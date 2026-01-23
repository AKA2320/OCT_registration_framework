use std::fs::{File};
use std::io::{Read, Seek, SeekFrom};
use bytemuck;
use rustfft::{FftPlanner, num_complex::Complex};
use ndarray_interp::interp1d::{Linear, Interp1DBuilder};
use ndarray::{Array2, Axis, s, ArrayView2, ArrayView1};

pub fn compute_fft_magnitude(input_spectrogram_linear: ArrayView2<f32>, fft_size: usize) -> Array2<f32> {
    let (rows, _) = input_spectrogram_linear.dim();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut magnitude = Array2::<f32>::zeros((rows, fft_size)); // Initialization here

    for (in_row, mut out_row) in input_spectrogram_linear.axis_iter(Axis(0)).zip(magnitude.axis_iter_mut(Axis(0))) {
        let mut buffer: Vec<Complex<f32>> = vec![Complex::default(); fft_size];
        for (i, &val) in in_row.iter().enumerate().take(fft_size) {
            buffer[i] = Complex::new(val, 0.0);
        }
        fft.process(&mut buffer);
        for (i, spec_val) in buffer.iter().enumerate() {
            out_row[i] = spec_val.norm(); // Gets the absolute value
        }
    }
    magnitude
}

pub fn interpolate_spectrogram(
    input_spectrogram: ArrayView2<f32>,
    k_raw: ArrayView1<f32>,
    k_linear: ArrayView1<f32>, 
    ) -> Array2<f32> {
    let (rows, _) = input_spectrogram.dim();
    let mut spectrogram_linear = Array2::zeros((rows, k_linear.len()));
    spectrogram_linear
        .axis_iter_mut(Axis(0))
        .zip(input_spectrogram.axis_iter(Axis(0)))
        .for_each(|(mut out_row, spec_row)| {
            let interpolator = Interp1DBuilder::new(spec_row)
                .x(k_raw)
                .strategy(Linear::new().extrapolate(true))
                .build()
                .unwrap();
            out_row.assign(&interpolator.interp_array(&k_linear).unwrap());
        });
    spectrogram_linear
}

pub fn load_spectrogram(
    path: String,
    buffer_lines: u64, 
    spectrometer_pixels: u64, 
    flip_spectrum: bool, 
    ) -> Array2<f32>{
    let mut file = File::open(path).expect("File not found");
    let file_size = file.metadata().expect("Error loading file metadata, check if corrupted").len();
    let expected_bytes = buffer_lines * spectrometer_pixels * 2;
    let header_size = file_size - expected_bytes;
    file.seek(SeekFrom::Start(header_size)).unwrap();
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let raw_u16: &[u16] = bytemuck::cast_slice(&buffer);
    let data: Vec<f32> = raw_u16.iter().map(|&x| x as f32).collect();

    let mut spectrogram = Array2::from_shape_vec((buffer_lines as usize, spectrometer_pixels as usize), data).unwrap();
    spectrogram = spectrogram.clone() - spectrogram.mean_axis(Axis(0)).unwrap();
    if flip_spectrum{
        spectrogram = spectrogram.slice(s![.., ..;-1]).to_owned();
    }
    spectrogram
}

pub fn process_single_bin(
    path: String,
    buffer_lines: u64, 
    spectrometer_pixels: u64, 
    flip_spectrum: bool, 
    k_raw: ArrayView1<f32>,
    k_linear: ArrayView1<f32>,
    fft_size: usize,
    default_shift_value: u64)
    -> (Array2<f32>, Array2<f32>) {
    let spectrogram = load_spectrogram(path, buffer_lines, spectrometer_pixels, flip_spectrum);
    let spectrogram_linear = interpolate_spectrogram(spectrogram.view(), k_raw, k_linear);
    let magnitude = compute_fft_magnitude(spectrogram_linear.view(), fft_size);

    let crop_depth = magnitude.ncols() / 2;
    let image_data = magnitude.slice(s![.., ..crop_depth]).to_owned();
    let (frame1, frame2) = reconstruct_frames(image_data, default_shift_value);
    (frame1, frame2)
}

pub fn reconstruct_frames(
    image_data: Array2<f32>,
    default_shift_value: u64
    ) -> (Array2<f32>, Array2<f32>){
    let shift = default_shift_value as usize;
    let image_data = image_data.slice(s![shift.., ..]);

    let mid_crop = image_data.nrows() / 2;
    let start_idx = image_data.nrows() - mid_crop;
    let mut frame_1_raw: ArrayView2<f32> = image_data.slice(s![..mid_crop, ..]);
    let mut frame_2_raw: ArrayView2<f32> = image_data.slice(s![start_idx.., ..]);

    frame_2_raw.invert_axis(Axis(0));
    // frame_1_raw.t().invert_axis(Axis(0));
    // frame_2_raw.t().invert_axis(Axis(0));
    frame_1_raw.swap_axes(0, 1);
    frame_1_raw.invert_axis(Axis(0));
    frame_2_raw.swap_axes(0, 1);
    frame_2_raw.invert_axis(Axis(0));

    (frame_1_raw.to_owned(), frame_2_raw.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{metadata};
    use ndarray::{Array1};

    #[test]
    fn check_reconstruct(){
        let buffer_lines: u64 = 1030;
        let spectrometer_pixels: u64 = 2048;
        let flip_spectrum = true;
        let path: String = "test_bin_file.bin".to_string();
        let k_raw: Array1<f32> = Array1::range(0., 2048., 1.);
        let k_linear: Array1<f32> = Array1::range(0., 2048., 1.);
        let fft_size: usize = 4096;
        let default_shift_value: u64 = 83;

        let (frame1, frame2) = process_single_bin(path, buffer_lines, spectrometer_pixels, flip_spectrum, k_raw.view(), k_linear.view(), fft_size, default_shift_value);
        println!("Shape: {:?}, {:?}", frame1.shape(), frame2.shape());
        assert_eq!(frame1.shape(), frame2.shape());
        assert_eq!(frame1.shape(), &[spectrometer_pixels as usize, 473_usize]);
    }    

    #[test]
    fn check_fft_magnitude_output(){
        let buffer_lines: u64 = 1030;
        let spectrometer_pixels: u64 = 2048;
        let flip_spectrum = true;
        let path: String = "test_bin_file.bin".to_string();
        let spectrogram = load_spectrogram(path, buffer_lines, spectrometer_pixels, flip_spectrum);

        let k_raw: Array1<f32> = Array1::range(0., 2048., 1.);
        let k_linear: Array1<f32> = Array1::range(0., 2048., 1.);
        let fft_size: usize = 4096;
        let spectrogram_linear = interpolate_spectrogram(spectrogram.view(), k_raw.view(), k_linear.view());

        let magnitude = compute_fft_magnitude(spectrogram_linear.view(), fft_size);
        println!("Shape: {:?}", magnitude.shape());
        assert_eq!(magnitude.shape(), &[buffer_lines as usize, fft_size as usize])
    }

    #[test]
    fn check_load_spectrogram(){
        let buffer_lines: u64 = 1030;
        let spectrometer_pixels: u64 = 2048;
        let flip_spectrum = true;
        let path: String = "test_bin_file.bin".to_string();
        let spectrogram = load_spectrogram(path, buffer_lines, spectrometer_pixels, flip_spectrum);

        println!("Shape: {:?}", spectrogram.shape());
        assert_eq!(spectrogram.shape(), &[1030,2048])
    }

    #[test]
    fn check_file_size(){
        let file_size = metadata("test_bin_file.bin").expect("File not found").len();
        let buffer_lines = 1030;
        let spectrometer_pixels = 2048;
        let expected_bytes = buffer_lines * spectrometer_pixels *2;
        let header_size = file_size - expected_bytes;
        println!("Size: {}", header_size);
        assert_eq!(header_size, 8_u64);
    }

}