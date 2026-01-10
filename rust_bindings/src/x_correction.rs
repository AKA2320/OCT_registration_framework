use crate::utility::min_max;
use ndarray::{Array2, ArrayView2, ArrayView3, Axis, concatenate, s};
use std::cmp::max;
use tch::{CModule, Device, IValue, Kind, Tensor};

pub fn compute_x_correction_pair(
    model_x: &CModule,
    static_data: Array2<f32>,
    moving_data: Array2<f32>,
    cells_coords_array: ArrayView2<usize>,
    enface_extraction_rows: &[usize],
) -> (f32, f32) {
    let mut cell_warps: Vec<(f32, f32)> = Vec::with_capacity(cells_coords_array.len());
    for coords_up_down in cells_coords_array.rows() {
        let (up_x, down_x) = (coords_up_down[0], coords_up_down[1]);
        let stat_frame = static_data.slice(s![up_x..down_x, ..]);
        let mov_frame = moving_data.slice(s![up_x..down_x, ..]);
        let (temp_cell_shift, inv_temp_cell_shift) = infer_x_translation(
            model_x,
            stat_frame.to_owned(),
            mov_frame.to_owned(),
            Device::Cpu,
        );
        let error_cell = (temp_cell_shift + inv_temp_cell_shift).abs();
        cell_warps.push((error_cell, temp_cell_shift));
    }

    let mut enface_warps: Vec<(f32, f32)> = Vec::with_capacity(enface_extraction_rows.len());
    for enf_val in enface_extraction_rows.iter() {
        let bottom_row = enf_val.saturating_sub(32);
        let stat_frame = static_data.slice(s![bottom_row..enf_val + 32, ..]);
        let mov_frame = moving_data.slice(s![bottom_row..enf_val + 32, ..]);
        let (temp_enface_shift, inv_temp_enface_shift) = infer_x_translation(
            model_x,
            stat_frame.to_owned(),
            mov_frame.to_owned(),
            Device::Cpu,
        );
        let error_enface = (temp_enface_shift + inv_temp_enface_shift).abs();
        enface_warps.push((error_enface, temp_enface_shift));
    }
    cell_warps.extend(enface_warps);
    let mut all_warps = cell_warps;
    all_warps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (all_warps[0].1, 0.0)
}

pub fn infer_x_translation(
    model: &CModule,
    static_arr: Array2<f32>,
    moving_arr: Array2<f32>,
    device: Device,
) -> (f32, f32) {
    let inp_static_arr = crop_or_pad(min_max(static_arr)).insert_axis(Axis(0));
    let inp_moving_arr = crop_or_pad(min_max(moving_arr)).insert_axis(Axis(0));

    let input_pair =
        concat_to_tensor(inp_static_arr.view(), inp_moving_arr.view()).to_device(device);
    let result: IValue = tch::no_grad(|| {
        model
            .forward_is(&[IValue::Tensor(input_pair)])
            .expect("Forward failed")
    });

    let input_pair_rev =
        concat_to_tensor(inp_moving_arr.view(), inp_static_arr.view()).to_device(device);
    let result_rev: IValue = tch::no_grad(|| {
        model
            .forward_is(&[IValue::Tensor(input_pair_rev)])
            .expect("Forward failed")
    });
    (extract_shift_val(result), extract_shift_val(result_rev))
}

pub fn crop_or_pad(mut arr: Array2<f32>) -> Array2<f32> {
    let target_height = 64;
    let target_width = 416;

    let (current_height, current_width): (usize, usize) = (arr.shape()[0], arr.shape()[1]);
    // PADDING LOGIC
    if (target_height > current_height) | (target_width > current_width) {
        let pad_top: usize = target_height.saturating_sub(current_height) / 2;
        let pad_left: usize = target_width.saturating_sub(current_width) / 2;
        let new_shape = (
            max(target_height, current_height),
            max(target_width, current_width),
        );

        let mut oversized_array = Array2::<f32>::zeros(new_shape);
        let row_start = pad_top;
        let row_end = pad_top + current_height;
        let col_start = pad_left;
        let col_end = pad_left + current_width;
        let target_slice = s![row_start..row_end, col_start..col_end];

        let mut center_view = oversized_array.slice_mut(target_slice);
        center_view.assign(&arr);

        arr = oversized_array.to_owned();
    }
    let (current_height, current_width): (usize, usize) = (arr.shape()[0], arr.shape()[1]);
    // CROPPING LOGIC
    if (target_height < current_height) | (target_width < current_width) {
        let crop_top: usize = current_height.saturating_sub(target_height) / 2;
        let crop_left: usize = current_width.saturating_sub(target_width) / 2;

        let row_start = crop_top;
        let row_end = crop_top + target_height;
        let col_start = crop_left;
        let col_end = crop_left + target_width;
        let target_slice = s![row_start..row_end, col_start..col_end];

        arr = arr.slice(target_slice).to_owned();
    }
    arr
}

pub fn load_model(path_model: &str) -> CModule {
    // let device = Device::Cpu;
    let mut model = CModule::load_on_device(path_model, Device::Cpu).expect("Failed to load model");
    model.set_eval();
    model
}

pub fn concat_to_tensor<'a>(arr1: ArrayView3<'a, f32>, arr2: ArrayView3<'a, f32>) -> Tensor {
    let concatenated = concatenate(Axis(0), &[arr1, arr2]).unwrap();
    let shape: Vec<i64> = concatenated.shape().iter().map(|&dim| dim as i64).collect();
    let data: Vec<f32> = concatenated.into_iter().collect();
    Tensor::from_slice(&data)
        .reshape(&shape)
        .to_kind(Kind::Double)
        .unsqueeze(0)
        .to_device(Device::Cpu)
}

pub fn extract_shift_val(result: IValue) -> f32 {
    let output: Tensor = if let IValue::Tuple(mut outputs) = result {
        let t1: Tensor = if let IValue::Tensor(temp_t1) = outputs.pop().unwrap() {
            temp_t1
        } else {
            panic!("Pop failed, vec was empty");
        };
        t1
    } else {
        panic!("something wrong with the output of the model");
    };
    let val: f32 = output.double_value(&[0, 0]) as f32;
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3, s};
    use ndarray_npy::NpzReader;
    use std::fs::File;

    #[test]
    fn check_x_correct_pair() {
        // let device = Device::Cpu;
        let model = load_model("../models/transmorph_lateral_X_translation.pt");
        let f = File::open("test_rust_arrays.npz").expect("Npz not found");
        let mut npz = NpzReader::new(f).expect("Couldnt read");

        let static_arr: Array3<f32> = npz.by_name("static.npy").expect("Couldnt Static");
        let moving_arr: Array3<f32> = npz.by_name("moving.npy").expect("Couldnt moving");

        let cells_coords_temp: Array2<u32> = npz
            .by_name("cells_coords.npy")
            .expect("Couldnt cells_coords");
        let cells_coords: Array2<usize> = cells_coords_temp.mapv(|x| x as usize);

        let enface_extraction_rows: Array1<i64> = npz
            .by_name("enface_extraction_rows.npy")
            .expect("Couldnt enface_extraction_rows");
        let enface_extraction_rows_vec: Vec<usize> = enface_extraction_rows
            .to_vec()
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let indices: Array1<i64> = npz.by_name("indices.npy").expect("Couldnt indices");

        let _transforms: Vec<(f32, f32)> = (0..indices.len())
            .into_iter()
            .map(|idx| {
                compute_x_correction_pair(
                    &model,
                    static_arr.slice(s![idx, .., ..]).to_owned(),
                    moving_arr.slice(s![idx, .., ..]).to_owned(),
                    cells_coords.view(),
                    &enface_extraction_rows_vec,
                )
            })
            .collect();
    }

    #[test]
    fn check_crop_or_pad() {
        let test_rows = &[47, 50, 64, 10, 151];
        let test_cols = &[9, 50, 416, 553, 5];
        for i in 0..test_rows.len() {
            let mut array1 = Array2::<f32>::ones((test_rows[i], test_cols[i]));
            array1 = crop_or_pad(array1);
            assert_eq!(
                array1.shape(),
                &[64 as usize, 416 as usize],
                "Shape mismatch at index {}. Expected [{}, {}], got {:?}",
                i,
                64,
                416,
                array1.shape()
            );
        }
    }

    #[test]
    fn torch_test() {
        let device = Device::Cpu;
        let model = load_model("../models/transmorph_lateral_X_translation.pt");

        let mut array1: Array2<f32> = Array2::<f32>::zeros((74, 416));
        array1.slice_mut(s![.., 300..350]).fill(20.0);

        let mut array2: Array2<f32> = Array2::<f32>::zeros((60, 416));
        array2.slice_mut(s![.., 304..354]).fill(5.0);

        let (val1, val2) = infer_x_translation(&model, array1, array2, device);

        println!("Val1: {} \n Val2: {}", val1, val2);
        assert!(val1 + val2 < 0.2);
    }
}
