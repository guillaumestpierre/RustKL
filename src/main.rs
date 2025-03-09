#![deny(clippy::unwrap_used)]
#![deny(clippy::except_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]
use image::{GenericImageView, Pixel};
use nalgebra::{DMatrix, DVector};
use std::fs;
use std::path::Path;
use std::io;

fn main() -> std::io::Result<()>{
    // Get image path
    println!("Enter the path to the original image.");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read from stdin");
    let mut image_path_str = input.trim().to_string();

    while !Path::new(&image_path_str).exists() {
        println!("Incorrect path, please enter another path.");
        image_path_str.clear();
        io::stdin().read_line(&mut image_path_str).expect("Failed to read from stdin");
        image_path_str = image_path_str.trim().to_string();
    }
    
    let image_path = Path::new(&image_path_str);

    // Loading image
    let img = image::open(image_path).expect("Could not load the image !");
    let (width, height) = img.dimensions();
    let nb_pixels = (width * height) as f64;

    // Stocking pixel values in matrix
    let mut image_matrix = DMatrix::<f64>::zeros((height * width) as usize, 3);
    for (x, y, pixel) in img.pixels() {
        let index = (y * width + x) as usize;
        let rgb = pixel.to_rgb();
        image_matrix[(index, 0)] = rgb[0] as f64;
        image_matrix[(index, 1)] = rgb[1] as f64;
        image_matrix[(index, 2)] = rgb[2] as f64;
    }

    // Mean calculation
    let moy_r = image_matrix.column(0).sum() / nb_pixels;
    let moy_g = image_matrix.column(1).sum() / nb_pixels;
    let moy_b = image_matrix.column(2).sum() / nb_pixels;
    println!("Mean R: {:.2}, G: {:.2}, B: {:.2}", moy_r, moy_g, moy_b);

    // Keeping a copy of the image
    let image_matrix_original = image_matrix.clone();

    // Data centering
    for i in 0..image_matrix.nrows() {
        image_matrix[(i, 0)] -= moy_r;
        image_matrix[(i, 1)] -= moy_g;
        image_matrix[(i, 2)] -= moy_b;
    }

    // Calculating cov : (X^T * X) / N
    let cov_rgb = (image_matrix.transpose() * &image_matrix) / nb_pixels;
    println!("Cov matrix:\n{}", cov_rgb);

    // Eig values and vectors
    let eig = cov_rgb.symmetric_eigen();
    println!("Eigenvalues:\n{}", eig.eigenvalues);
    println!("Eigenvectors:\n{}", eig.eigenvectors);

    // Deleting components
    let mut diag_sans_axe0 = DMatrix::<f64>::identity(3, 3);
    diag_sans_axe0[(0, 0)] = 0.0;

    let mut diag_sans_axe1 = DMatrix::<f64>::identity(3, 3);
    diag_sans_axe1[(1, 1)] = 0.0;

    let mut diag_sans_axe2 = DMatrix::<f64>::identity(3, 3);
    diag_sans_axe2[(2, 2)] = 0.0;

    // Projection matrix
    let proj_sans_axe0 = &eig.eigenvectors * &diag_sans_axe0 * &eig.eigenvectors.transpose();
    let proj_sans_axe1 = &eig.eigenvectors * &diag_sans_axe1 * &eig.eigenvectors.transpose();
    let proj_sans_axe2 = &eig.eigenvectors * &diag_sans_axe2 * &eig.eigenvectors.transpose();

    // Stocking the results in matrix
    let mut image_RGB_sans_axe0 = DMatrix::<f64>::zeros(image_matrix.nrows(), 3);
    let mut image_RGB_sans_axe1 = DMatrix::<f64>::zeros(image_matrix.nrows(), 3);
    let mut image_RGB_sans_axe2 = DMatrix::<f64>::zeros(image_matrix.nrows(), 3);

    let vec_moy = DVector::<f64>::from_vec(vec![moy_r, moy_g, moy_b]);

    for i in 0..image_matrix_original.nrows() {
        // Creating a vector for each pixel
        let vec_original = DVector::<f64>::from_vec(vec![
            image_matrix_original[(i, 0)],
            image_matrix_original[(i, 1)],
            image_matrix_original[(i, 2)],
        ]);
        
        // Centering the data
        let vec_centered = &vec_original - &vec_moy;
        
        // Apply projection
        let proj_vec0 = &proj_sans_axe0 * &vec_centered;
        let proj_vec1 = &proj_sans_axe1 * &vec_centered;
        let proj_vec2 = &proj_sans_axe2 * &vec_centered;
        
        // Reconstruct
        let reconstructed0 = &proj_vec0 + &vec_moy;
        let reconstructed1 = &proj_vec1 + &vec_moy;
        let reconstructed2 = &proj_vec2 + &vec_moy;
        
        // Stocking the values with clamping
        image_RGB_sans_axe0[(i, 0)] = reconstructed0[0].clamp(0.0, 255.0);
        image_RGB_sans_axe0[(i, 1)] = reconstructed0[1].clamp(0.0, 255.0);
        image_RGB_sans_axe0[(i, 2)] = reconstructed0[2].clamp(0.0, 255.0);
        
        image_RGB_sans_axe1[(i, 0)] = reconstructed1[0].clamp(0.0, 255.0);
        image_RGB_sans_axe1[(i, 1)] = reconstructed1[1].clamp(0.0, 255.0);
        image_RGB_sans_axe1[(i, 2)] = reconstructed1[2].clamp(0.0, 255.0);
        
        image_RGB_sans_axe2[(i, 0)] = reconstructed2[0].clamp(0.0, 255.0);
        image_RGB_sans_axe2[(i, 1)] = reconstructed2[1].clamp(0.0, 255.0);
        image_RGB_sans_axe2[(i, 2)] = reconstructed2[2].clamp(0.0, 255.0);
    }

    //Saving reconstructed images
    let (mut img_reconstructed0, mut img_reconstructed1, mut img_reconstructed2) = (image::RgbImage::new(width, height), image::RgbImage::new(width, height),image::RgbImage::new(width, height));
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel0 = image::Rgb([
                image_RGB_sans_axe0[(index, 0)] as u8,
                image_RGB_sans_axe0[(index, 1)] as u8,
                image_RGB_sans_axe0[(index, 2)] as u8,
            ]);
            let pixel1 = image::Rgb([
                image_RGB_sans_axe1[(index, 0)] as u8,
                image_RGB_sans_axe1[(index, 1)] as u8,
                image_RGB_sans_axe1[(index, 2)] as u8,
            ]);
            let pixel2 = image::Rgb([
                image_RGB_sans_axe2[(index, 0)] as u8,
                image_RGB_sans_axe2[(index, 1)] as u8,
                image_RGB_sans_axe2[(index, 2)] as u8,
            ]);
            img_reconstructed0.put_pixel(x, y, pixel0);
            img_reconstructed1.put_pixel(x, y, pixel1);
            img_reconstructed2.put_pixel(x, y, pixel2);
        }
    }
    // Saving
    println!("The 3 modified images are being saved to the /result folder.");
    if !Path::new("./result").exists() {
        fs::create_dir("./result")?;
    }    
    img_reconstructed0.save("./result/reconstructed0.png").expect("Could not save the image !");
    img_reconstructed1.save("./result/reconstructed1.png").expect("Could not save the image !");
    img_reconstructed2.save("./result/reconstructed2.png").expect("Could not save the image !");

    Ok(())

}
