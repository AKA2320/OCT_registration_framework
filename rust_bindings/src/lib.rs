// #![allow(unused, dead_code, unused_variables, unused_imports)]

use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayView3, Axis, s};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
mod flat_minimization;
mod utility;
mod x_correction;
mod y_minimization;

use flat_minimization::*;
use utility::*;
use x_correction::*;
use y_minimization::*;

#[pyfunction]
fn run_y_correction_compute_rust(
    py: Python,
    stat_data: PyReadonlyArray2<f32>,
    mov_data: PyReadonlyArray3<f32>,
) -> PyResult<Vec<(f32, f32)>> {
    let static_image = ndarray_to_kornia_image(stat_data.as_array().to_owned()); // (m * n)
    let moving_data: Array3<f32> = mov_data.as_array().to_owned(); // (l * m * n)

    let transforms: Vec<(f32, f32)> = py.detach(|| {
        moving_data
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|slice1| compute_y_motion(&static_image, slice1.to_owned()))
            .collect()
    });
    Ok(transforms)
}

#[pyfunction]
fn run_flat_correction_compute_rust(
    py: Python,
    stat_data: PyReadonlyArray2<f32>,
    mov_data: PyReadonlyArray3<f32>,
) -> PyResult<Vec<(f32, f32)>> {
    let static_image = ndarray_to_kornia_image(stat_data.as_array().to_owned()); // (m * n)
    let moving_data: Array3<f32> = mov_data.as_array().to_owned(); // (l * m * n)

    let transforms: Vec<(f32, f32)> = py.detach(|| {
        moving_data
            .axis_iter(Axis(2))
            .into_par_iter()
            .map(|slice1| compute_flat_motion(&static_image, slice1.to_owned()))
            .collect()
    });
    Ok(transforms)
}

#[pyfunction]
fn run_x_correction_compute_rust(
    py: Python,
    stat_data: PyReadonlyArray3<f32>,
    mov_data: PyReadonlyArray3<f32>,
    indices: Vec<usize>,
    enface_extraction_rows: Vec<usize>,
    cells_coords: PyReadonlyArray2<u32>,
    valid_args: Vec<usize>,
    model_path: &str,
) -> PyResult<Vec<(f32, f32)>> {
    let static_data: ArrayView3<f32> = stat_data.as_array(); // (l * m * n)
    let moving_data: ArrayView3<f32> = mov_data.as_array(); // (l * m * n)
    let cells_coords_array: Array2<usize> = cells_coords.as_array().to_owned().mapv(|x| x as usize);
    let model = load_model(model_path);

    let transforms: Vec<(f32, f32)> = py.detach(|| {
        moving_data
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(idx, slice_mov)| {
                if valid_args.contains(&indices[idx]) {
                    let slice_stat = static_data.slice(s![idx, .., ..]);
                    compute_x_correction_pair(
                        &model,
                        slice_stat.to_owned(),
                        slice_mov.to_owned(),
                        cells_coords_array.view(),
                        &enface_extraction_rows,
                    )
                } else {
                    (0.0, 0.0)
                }
            })
            .collect()
    });

    Ok(transforms)
}

#[pymodule]
fn rust_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_y_correction_compute_rust, m)?)?;
    m.add_function(wrap_pyfunction!(run_flat_correction_compute_rust, m)?)?;
    m.add_function(wrap_pyfunction!(run_x_correction_compute_rust, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array3, s};
    use std::time::Instant;

    #[test]
    fn run_y_correct() {
        let start_time = Instant::now();
        let mut array1: Array3<f32> = Array::<f32, _>::zeros((20, 100, 500));
        array1.slice_mut(s![10, 20..75, 300..350]).fill(1.0);
        let array1_image = ndarray_to_kornia_image(array1.slice(s![10, .., ..]).to_owned());

        let shifts: Vec<i32> = (-10..10).collect();

        let results: Vec<i32> = (0..array1.shape()[0])
            .into_par_iter()
            .map(|idx| {
                let x = shifts[idx];
                let mut temp_arr2 = array1.slice(s![idx, .., ..]).to_owned();
                temp_arr2
                    .slice_mut(s![(20 + x)..(75 + x), 300..350])
                    .fill(1.0);
                compute_y_motion(&array1_image, temp_arr2).1.round() as i32
            })
            .collect();

        for i in 0..results.len() {
            assert_eq!(-results[i], shifts[i]);
        }
        println!(
            "Compute Y motion function took: {:?} seconds",
            Instant::now().duration_since(start_time).as_secs()
        );
    }

    #[test]
    fn run_flatten() {
        let start_time = Instant::now();
        let mut array1: Array3<f32> = Array::<f32, _>::zeros((20, 100, 50));
        // array1.slice_mut(s![25, 20..75, 300..350]).fill(1.0);
        let shifts: Vec<i32> = (-25..25).collect();
        for idx in 0..shifts.len() {
            array1
                .slice_mut(s![.., (50 + shifts[idx])..(75 + shifts[idx]), idx])
                .fill(1.0);
        }
        let array1_image = ndarray_to_kornia_image(array1.slice(s![.., .., 25]).to_owned());

        // let shifts: Vec<i32> = (-25..25).collect();
        let mut results: Vec<i32> = vec![];
        // println!("{:?}", shifts);
        for idx in 0..array1.shape()[2] {
            // let x = shifts[idx];
            let temp_arr2 = array1.slice(s![.., .., idx]).to_owned();
            // temp_arr2.slice_mut(s![(30+x)..(65+x), 300..350]).fill(1.0);
            results.push(compute_flat_motion(&array1_image, temp_arr2).0.round() as i32);
        }

        for i in 0..results.len() {
            assert_eq!(-results[i], shifts[i]);
        }
        println!(
            "Compute flatten function took: {:?} seconds",
            Instant::now().duration_since(start_time).as_secs()
        );
    }
}
