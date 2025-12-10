use crate::utility::{ncc, ndarray_to_kornia_image, warp_image_kornia};
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::brent::BrentOpt;
use kornia_rs::image::Image;
use ndarray::{Array2, Axis};

pub fn err_func_flat_motion(
    shift: f32,
    image_x: &Image<f32, 1>,
    image_y: &Image<f32, 1>,
    past_shift: f32,
) -> f32 {
    let warped_x = warp_image_kornia(image_x, (-shift - past_shift, 0.0));
    let warped_y = warp_image_kornia(image_y, (shift + past_shift, 0.0));

    let corr_err = ncc(
        warped_x.data.remove_axis(Axis(2)).view(),
        warped_y.data.remove_axis(Axis(2)).view(),
    )
    .unwrap();

    1.0 - corr_err
}

#[derive(Clone)]
struct FlatMotionErrorCost<'a> {
    image_x: &'a Image<f32, 1>,
    image_y: &'a Image<f32, 1>,
    past_shift: f32,
}

impl<'a> CostFunction for FlatMotionErrorCost<'a> {
    type Param = f32;
    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let shift = param;
        Ok(err_func_flat_motion(
            *shift,
            self.image_x,
            self.image_y,
            self.past_shift,
        ))
    }
}

pub fn compute_flat_motion(static_image_arr1: &Image<f32, 1>, arr2: Array2<f32>) -> (f32, f32) {
    // let static_image_arr1 = ndarray_to_kornia_image(arr1);
    let moving_image_arr2 = ndarray_to_kornia_image(arr2);
    let mut past_shift: f32 = 0.0;
    let shift_threshold: f32 = 0.05;

    for _ in 0..10 {
        let cost: FlatMotionErrorCost<'_> = FlatMotionErrorCost {
            image_x: static_image_arr1,
            image_y: &moving_image_arr2,
            past_shift,
        };

        // let p0 = vec![0.0_f32];
        let solver = BrentOpt::new(-10.0_f32, 10.0_f32);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(2000)) // Limit iterations per loop
            .run();

        match res {
            Ok(result) => {
                let state = result.state();
                let move_val = state.best_param.unwrap();
                if move_val.abs() < shift_threshold {
                    break;
                }
                past_shift += move_val;
            }
            Err(_) => {
                return (0.0, 0.0); // Identity matrix on error
            }
        }
    }

    let tx = past_shift * 2.0;
    let ty = 0.0;
    (tx, ty)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, s};
    // use image::{ImageBuffer, Luma};
    use super::*;

    #[test]
    fn shift_transfrom() {
        let mut array1: Array2<f32> = Array::<f32, _>::zeros((50, 1000));
        array1.slice_mut(s![15..20, 500..550]).fill(1.0);
        let array1_image = ndarray_to_kornia_image(array1);

        let mut array2: Array2<f32> = Array::<f32, _>::zeros((50, 1000));
        array2.slice_mut(s![15..20, 517..567]).fill(1.0);

        // let (h, w1) = array1.dim();
        // let buffer1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(
        // w1 as u32,
        // h as u32,
        // array1.mapv(|v| (v * 255.0).clamp(0.0, 255.0) as u8).as_slice().unwrap().to_vec(),
        // )
        // .unwrap();
        // buffer1.save("test_array_big1.png").unwrap();

        let res_transfrom: (f32, f32) = compute_flat_motion(&array1_image, array2);
        println!("{:?}", res_transfrom);
        assert!(res_transfrom.0.round() as i32 == -17);
    }

    #[test]
    fn no_shift_transfrom() {
        let mut array1: Array2<f32> = Array::<f32, _>::zeros((50, 1000));
        array1.slice_mut(s![20..25, 500..550]).fill(1.0);
        let array1_image = ndarray_to_kornia_image(array1);

        let mut array2: Array2<f32> = Array::<f32, _>::zeros((50, 1000));
        array2.slice_mut(s![20..25, 500..550]).fill(1.0);

        let res_transfrom: (f32, f32) = compute_flat_motion(&array1_image, array2);
        // println!("{:?}", res_transfrom);
        assert!(res_transfrom.0.round() as i32 == 0);
    }
}
