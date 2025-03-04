# RustKL

Image Compression and Reconstruction using the KL transform in Rust.
It currently has a considerable luminosity loss during the transformation.
I'm working on it o7

# Installation

```
#Clone the repository
git clone https://github.com/guillaumestpierre/RustKL.git
```

# Usage
```
#Navigate to the project directory
cd RustKL

#Run the application
cargo run --release
```

# Dependencies
This project uses the following Rust crates:

- nalgebra - for mathematical operations
- image - for image processing

# Example
- input :
  ![Example Image](https://github.com/guillaumestpierre/RustKL/tree/main/data/kodim13.png)
  
- output :
![Example Image](https://github.com/guillaumestpierre/RustKL/tree/main/result/reconstructed0.png)
![Example Image](https://github.com/guillaumestpierre/RustKL/tree/main/result/reconstructed1.png)
![Example Image](https://github.com/guillaumestpierre/RustKL/tree/main/result/reconstructed2.png)
