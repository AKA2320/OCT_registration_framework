use kornia_rs::image::{Image, ImageSize};
use kornia_rs::interpolation::InterpolationMode;
use kornia_rs::warp::warp_affine;
use ndarray::{Array2, ArrayView2};

pub fn min_max(arr: Array2<f32>) -> Array2<f32> {
    let (min_val, max_val) = arr
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });

    if max_val == min_val {
        Array2::<f32>::zeros((arr.shape()[0], arr.shape()[1]))
    } else {
        arr.map(|&x| (x - min_val) / (max_val - min_val))
    }
}

pub fn ndarray_to_kornia_image<T: Copy + Send + Sync>(arr: Array2<T>) -> Image<T, 1> {
    let height = arr.nrows();
    let width = arr.ncols();
    Image::new(ImageSize { height, width }, arr.into_raw_vec())
        .expect("Failed to create kornia Image — invalid dimensions")
}

pub fn warp_image_kornia(img: &Image<f32, 1>, shifts: (f32, f32)) -> Image<f32, 1> {
    let m: (f32, f32, f32, f32, f32, f32) = (1.0, 0.0, shifts.0, 0.0, 1.0, shifts.1);
    let (height, width) = (img.height(), img.width());

    warp_affine(
        img,
        m,
        ImageSize { height, width },
        InterpolationMode::Bilinear,
    )
    .expect("Kornia Warping failed")
}

#[derive(Debug, PartialEq)]
pub enum NccError {
    DimMisMatch,
}

pub fn ncc(a1: ArrayView2<f32>, a2: ArrayView2<f32>) -> Result<f32, NccError> {
    if a1.dim() != a2.dim() {
        return Err(NccError::DimMisMatch);
    }

    let m1 = a1.mean().unwrap_or(0.0); // 50,500
    let m2 = a2.mean().unwrap_or(0.0); // 50,500

    let (numerator, var_sum1, var_sum2) =
        a1.iter()
            .zip(a2.iter())
            .fold((0.0_f32, 0.0_f32, 0.0_f32), |(num, s1, s2), (&x, &y)| {
                let d1 = x - m1;
                let d2 = y - m2;
                ((num + (d1 * d2)), (s1 + d1 * d1), (s2 + d2 * d2))
            });
    let sigma1 = var_sum1.sqrt();
    let sigma2 = var_sum2.sqrt();
    let denominator = sigma1 * sigma2;

    if denominator != 0.0 {
        Ok(numerator / denominator)
    } else if (sigma1.abs() < 1e-6) & (sigma2.abs() < 1e-6) {
        Ok(1.0)
    } else {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};
    use ndarray::{Array, array};

    #[test]
    fn vaild_ncc() {
        let array1 = Array::<f32, _>::ones((500, 500));
        let array2 = Array::<f32, _>::ones((500, 500));

        assert_eq!(ncc(array1.view(), array2.view(),).unwrap(), 1.0_f32);

        let array1: Array2<f32> = array![[1.0_f32, 2.0_f32, 3.0_f32], [1.0_f32, 1.0_f32, 1.0_f32]];
        let array2: Array2<f32> = array![[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]];
        assert!(ncc(array1.view(), array2.view(),).unwrap() < 0.66_f32);
        assert!(ncc(array1.view(), array2.view(),).unwrap() > 0.64_f32);
    }

    #[test]
    fn wrong_dim() {
        let array1 = Array::<f32, _>::ones((5, 5));
        let array3 = Array::<f32, _>::ones((2, 5));
        let error = ncc(array1.view(), array3.view()).expect_err("Should have failed");

        assert_eq!(error, NccError::DimMisMatch);
    }

    #[test]
    fn ndarray_to_kornia_check() {
        let array1 = Array::<f64, _>::ones((523, 500));
        let (row_arr, col_arr) = { (array1.nrows(), array1.ncols()) };
        let kornia_image = ndarray_to_kornia_image(array1);
        println!(
            "{}, {}, {}, {}",
            row_arr,
            col_arr,
            kornia_image.rows(),
            kornia_image.cols()
        );
        assert_eq!(
            (row_arr, col_arr),
            (kornia_image.rows(), kornia_image.cols())
        );
    }

    #[test]
    fn warp_check() {
        let array1 = Array::<f32, _>::ones((523, 500));
        let arr_shape = array1.dim();
        let kornia_image = ndarray_to_kornia_image(array1);
        let warped = warp_image_kornia(&kornia_image, (2.0, 2.0));
        assert_eq!(arr_shape, (warped.height(), warped.width()));
    }

    #[test]
    fn plot_warp_check() {
        let arr1: Array2<f32> = array![[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]];
        let (h, w1) = arr1.dim();

        let buffer1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(
            w1 as u32,
            h as u32,
            arr1.mapv(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                .as_slice()
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        buffer1.save("test_img1.png").unwrap();

        let arr1: Array2<f32> = array![[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]];
        let (h, w1) = arr1.dim();
        let temp_image = ndarray_to_kornia_image(arr1);
        let warped_image = warp_image_kornia(&temp_image, (1.0, 1.0)).data;

        let buffer1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(
            w1 as u32,
            h as u32,
            warped_image
                .mapv(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                .as_slice()
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        buffer1.save("test_warped_img1.png").unwrap();
    }
}
