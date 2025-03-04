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
    // Récupérer le path de l'image
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

    // Charger l'image
    let img = image::open(image_path).expect("Impossible de charger l'image");
    let (width, height) = img.dimensions();
    let nb_pixels = (width * height) as f64;

    // Stocker les valeurs des pixels sous forme de matrices
    let mut image_matrix = DMatrix::<f64>::zeros((height * width) as usize, 3);

    for (x, y, pixel) in img.pixels() {
        let index = (y * width + x) as usize;
        let rgb = pixel.to_rgb();
        image_matrix[(index, 0)] = rgb[0] as f64;
        image_matrix[(index, 1)] = rgb[1] as f64;
        image_matrix[(index, 2)] = rgb[2] as f64;
    }

    // Calcul des moyennes
    let moy_r = image_matrix.column(0).sum() / nb_pixels;
    let moy_g = image_matrix.column(1).sum() / nb_pixels;
    let moy_b = image_matrix.column(2).sum() / nb_pixels;

    println!("Moyenne R: {:.2}, G: {:.2}, B: {:.2}", moy_r, moy_g, moy_b);

    // Centrage des données
    for i in 0..image_matrix.nrows() {
        image_matrix[(i, 0)] -= moy_r;
        image_matrix[(i, 1)] -= moy_g;
        image_matrix[(i, 2)] -= moy_b;
    }

    // Calcul de la matrice de covariance : (X^T * X) / N
    let cov_rgb = (image_matrix.transpose() * &image_matrix) / nb_pixels;
    println!("Matrice de covariance:\n{}", cov_rgb);

    // Calcul des valeurs propres et vecteurs propres avec nalgebra
    let eig = cov_rgb.symmetric_eigen();
    println!("Valeurs propres:\n{}", eig.eigenvalues);
    println!("Vecteurs propres:\n{}", eig.eigenvectors);


    //Une composante sur trois retirée
    let trans_eig_vec = eig.eigenvectors.transpose();

    let mut eig_vec_sans_axe0 = trans_eig_vec.clone();
    eig_vec_sans_axe0.fill_row(0, 0.0);

    let mut eig_vec_sans_axe1 = trans_eig_vec.clone();
    eig_vec_sans_axe1.fill_row(1, 0.0);

    let mut eig_vec_sans_axe2 = trans_eig_vec.clone();
    eig_vec_sans_axe2.fill_row(2, 0.0);

    let mut image_KL_sans_axe0 = image_matrix.clone();
    let mut image_KL_sans_axe1 = image_matrix.clone();
    let mut image_KL_sans_axe2 = image_matrix.clone();

    let vec_moy = DVector::<f64>::from_vec(vec![moy_r, moy_g, moy_b]);

    for i in 0..image_matrix.nrows() {
        // Création du vecteur pour chaque pixel
        let vec_temp = DVector::<f64>::from_vec(vec![
            image_matrix[(i, 0)],
            image_matrix[(i, 1)],
            image_matrix[(i, 2)],
        ]);

        // Calcul de la soustraction du vecteur moyen
        let temp_minus_moy = &vec_temp - &vec_moy;

        // Calcul des nouvelles valeurs pour chaque image transformée
        let transformed_axe0 = &eig_vec_sans_axe0 * &temp_minus_moy;
        let transformed_axe1 = &eig_vec_sans_axe1 * &temp_minus_moy;
        let transformed_axe2 = &eig_vec_sans_axe2 * &temp_minus_moy;

        // Stockage dans les matrices de sortie
        image_KL_sans_axe0[(i, 0)] = transformed_axe0[0];
        image_KL_sans_axe0[(i, 1)] = transformed_axe0[1];
        image_KL_sans_axe0[(i, 2)] = transformed_axe0[2];

        image_KL_sans_axe1[(i, 0)] = transformed_axe1[0];
        image_KL_sans_axe1[(i, 1)] = transformed_axe1[1];
        image_KL_sans_axe1[(i, 2)] = transformed_axe1[2];

        image_KL_sans_axe2[(i, 0)] = transformed_axe2[0];
        image_KL_sans_axe2[(i, 1)] = transformed_axe2[1];
        image_KL_sans_axe2[(i, 2)] = transformed_axe2[2];
    }      

    //Reconstruction des images
    let inv_eigvec_sans_axe0 = eig_vec_sans_axe0.clone().pseudo_inverse(1e-10);
    let inv_eigvec_sans_axe1 = eig_vec_sans_axe1.clone().pseudo_inverse(1e-10);
    let inv_eigvec_sans_axe2 = eig_vec_sans_axe2.clone().pseudo_inverse(1e-10);

    if !inv_eigvec_sans_axe0.is_ok() || !inv_eigvec_sans_axe1.is_ok() || !inv_eigvec_sans_axe2.is_ok() {
        panic!("Une des matrices modifiées n'est pas inversible !");
    }

    let inv_eigvec_sans_axe0 = inv_eigvec_sans_axe0.unwrap();
    let inv_eigvec_sans_axe1 = inv_eigvec_sans_axe1.unwrap();
    let inv_eigvec_sans_axe2 = inv_eigvec_sans_axe2.unwrap();

    let mut image_RGB_sans_axe0 = image_matrix.clone();
    let mut image_RGB_sans_axe1 = image_matrix.clone();
    let mut image_RGB_sans_axe2 = image_matrix.clone();

    for i in 0..image_matrix.nrows() {
        // Création du vecteur pour chaque pixel
        let vec_temp_0 = DVector::<f64>::from_vec(vec![
            image_KL_sans_axe0[(i, 0)],
            image_KL_sans_axe0[(i, 1)],
            image_KL_sans_axe0[(i, 2)],
        ]);

        let vec_temp_1 = DVector::<f64>::from_vec(vec![
            image_KL_sans_axe1[(i, 0)],
            image_KL_sans_axe1[(i, 1)],
            image_KL_sans_axe1[(i, 2)],
        ]);

        let vec_temp_2 = DVector::<f64>::from_vec(vec![
            image_KL_sans_axe2[(i, 0)],
            image_KL_sans_axe2[(i, 1)],
            image_KL_sans_axe2[(i, 2)],
        ]);

        let reconstructed_axe0 = &inv_eigvec_sans_axe0 * vec_temp_0 + &vec_moy;
        let reconstructed_axe1 = &inv_eigvec_sans_axe1 * vec_temp_1 + &vec_moy;
        let reconstructed_axe2 = &inv_eigvec_sans_axe2 * vec_temp_2 + &vec_moy;

        // Stockage dans les matrices RGB reconstruites
        image_RGB_sans_axe0[(i, 0)] = reconstructed_axe0[0];
        image_RGB_sans_axe0[(i, 0)] = image_RGB_sans_axe0[(i, 0)].clamp(0.0, 255.0);
        image_RGB_sans_axe0[(i, 1)] = reconstructed_axe0[1];
        image_RGB_sans_axe0[(i, 1)] = image_RGB_sans_axe0[(i, 1)].clamp(0.0, 255.0);
        image_RGB_sans_axe0[(i, 2)] = reconstructed_axe0[2];
        image_RGB_sans_axe0[(i, 2)] = image_RGB_sans_axe0[(i, 2)].clamp(0.0, 255.0);

        image_RGB_sans_axe1[(i, 0)] = reconstructed_axe1[0];
        image_RGB_sans_axe1[(i, 0)] = image_RGB_sans_axe1[(i, 0)].clamp(0.0, 255.0);
        image_RGB_sans_axe1[(i, 1)] = reconstructed_axe1[1];
        image_RGB_sans_axe1[(i, 1)] = image_RGB_sans_axe1[(i, 1)].clamp(0.0, 255.0);
        image_RGB_sans_axe1[(i, 2)] = reconstructed_axe1[2];
        image_RGB_sans_axe1[(i, 2)] = image_RGB_sans_axe1[(i, 2)].clamp(0.0, 255.0);

        image_RGB_sans_axe2[(i, 0)] = reconstructed_axe2[0];
        image_RGB_sans_axe2[(i, 0)] = image_RGB_sans_axe2[(i, 0)].clamp(0.0, 255.0);
        image_RGB_sans_axe2[(i, 1)] = reconstructed_axe2[1];
        image_RGB_sans_axe2[(i, 1)] = image_RGB_sans_axe2[(i, 1)].clamp(0.0, 255.0);
        image_RGB_sans_axe2[(i, 2)] = reconstructed_axe2[2];
        image_RGB_sans_axe2[(i, 2)] = image_RGB_sans_axe2[(i, 2)].clamp(0.0, 255.0);
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

    // Sauvegarde
    println!("The 3 modified images are being saved to the /result folder.");
    if !Path::new("./result").exists() {
        fs::create_dir("./result")?;
    }    
    img_reconstructed0.save("./result/reconstructed0.png").expect("Impossible de sauvegarder l'image !");
    img_reconstructed1.save("./result/reconstructed1.png").expect("Impossible de sauvegarder l'image !");
    img_reconstructed2.save("./result/reconstructed2.png").expect("Impossible de sauvegarder l'image !");

    Ok(())

}
