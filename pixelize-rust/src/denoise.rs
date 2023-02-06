#![allow(non_snake_case)]
use std::{ops::Mul, };

//use image;
use std::collections::HashMap;
use onnxruntime::{
    ndarray::{Array, Ix4, Ix5, IxDyn, s, concatenate, Axis},
};

fn denoise_1024_(models: &mut HashMap<String, onnxruntime::session::Session<'_>>,
           frames:Array<f32, Ix5>, noise_level:f32) -> Array<f32, Ix4>{
    let block1 = models.get_mut(&("block1".to_string())).unwrap();
    let noise_arr : Array<f32, Ix4> = Array::ones((1,3,1024,1024)).mul(noise_level);
    let noise_arr = noise_arr.into_dyn();

    let a0 : Array<f32, Ix5> = frames.slice(s![..,0..3,..,..,..]).to_owned();
    let a0 = a0.into_shape((1,9,1024,1024)).unwrap().into_dyn();
    let a0:Array<f32, IxDyn> = concatenate![Axis(1), a0, noise_arr];
    let b0:Array<f32, Ix4> = frames.slice(s![..,1,..,..,..]).to_owned();
    let b0 = b0.into_dyn();
    let c0 = (*block1.run::<'_, '_, '_, f32, f32, IxDyn>(vec![a0, b0]).unwrap()[0]).to_owned();

    let a1 : Array<f32, Ix5> = frames.slice(s![..,1..4,..,..,..]).to_owned();
    let a1 = a1.into_shape((1,9,1024,1024)).unwrap().into_dyn();
    let a1:Array<f32, IxDyn> = concatenate![Axis(1), a1, noise_arr];
    let b1:Array<f32, Ix4> = frames.slice(s![..,1,..,..,..]).to_owned();
    let b1 = b1.into_dyn();
    let c1 = (*block1.run::<'_, '_, '_, f32, f32, IxDyn>(vec![a1, b1]).unwrap()[0]).to_owned();

    let a2 : Array<f32, Ix5> = frames.slice(s![..,2..5,..,..,..]).to_owned();
    let a2 = a2.into_shape((1,9,1024,1024)).unwrap().into_dyn();
    let a2:Array<f32, IxDyn> = concatenate![Axis(1), a2, noise_arr];
    let b2:Array<f32, Ix4> = frames.slice(s![..,1,..,..,..]).to_owned();
    let b2 = b2.into_dyn();
    let c2 = (*block1.run::<'_, '_, '_, f32, f32, IxDyn>(vec![a2, b2]).unwrap()[0]).to_owned();

    let block2 = models.get_mut(&("block2".to_string())).unwrap();
    let d:Array<f32, IxDyn> = concatenate![Axis(1), c0, c1, c2, noise_arr];
    let e:Array<f32, IxDyn> = frames.slice(s![..,2,..,..,..]).to_owned().into_dyn();
    let result = (*block2.run::<'_, '_, '_, f32, f32, IxDyn>(vec![d, e]).unwrap()[0]).to_owned();

    let output: Array<f32, Ix4> = result.into_dimensionality::<Ix4>().unwrap();
    output
}

pub fn denoise_1024(models: &mut HashMap<String, onnxruntime::session::Session<'_>>,
           arrs_all:Vec<(String, Array<f32, Ix4>, (u32,u32))>, noise_level:f32) -> Vec<(String, Array<f32, Ix4>, (u32,u32))>{
    let n_frames:usize = arrs_all.len();
    if n_frames >= 5 {
        let mut denoised_all: Vec<(String, Array<f32, Ix4>, (u32,u32))> = Vec::new();
        for i in 0..n_frames{
            // FIXME
            println!("Denoising {}", &arrs_all[i].0);
            let f0 = &arrs_all[if i<2 {0} else {i-2}].1;
            let f1 = &arrs_all[if i<1 {0} else {i-1}].1;
            let f2 = &arrs_all[i].1;
            let f3 = &arrs_all[if i>n_frames-2 {n_frames-1} else {i+1}].1;
            let f4 = &arrs_all[if i>n_frames-3 {n_frames-1} else {i+2}].1;
            let frames : Array<f32,Ix4> = ndarray::concatenate![Axis(0), f0.clone(), f1.clone(), f2.clone(), f3.clone(), f4.clone()];
            let frames : Array<f32,Ix5> = frames.insert_axis(Axis(0));
            let denoised = denoise_1024_(models, frames, noise_level);
            denoised_all.push( (arrs_all[i].0.clone(), denoised, arrs_all[i].2) );
        }
        denoised_all
    }
    else{
        arrs_all
    }
}


fn denoise_256_(models: &mut HashMap<String, onnxruntime::session::Session<'_>>,
           frames:Array<f32, Ix5>, noise_level:f32) -> Array<f32, Ix4>{
    let block1 = models.get_mut(&("block1_256".to_string())).unwrap();
    let noise_arr : Array<f32, Ix4> = Array::ones((1,3,256,256)).mul(noise_level);
    let noise_arr = noise_arr.into_dyn();

    let a0 : Array<f32, Ix5> = frames.slice(s![..,0..3,..,..,..]).to_owned();
    let a0 = a0.into_shape((1,9,256,256)).unwrap().into_dyn();
    let a0:Array<f32, IxDyn> = concatenate![Axis(1), a0, noise_arr];
    let b0:Array<f32, Ix4> = frames.slice(s![..,1,..,..,..]).to_owned();
    let b0 = b0.into_dyn();
    let c0 = (*block1.run::<'_, '_, '_, f32, f32, IxDyn>(vec![a0, b0]).unwrap()[0]).to_owned();

    let a1 : Array<f32, Ix5> = frames.slice(s![..,1..4,..,..,..]).to_owned();
    let a1 = a1.into_shape((1,9,256,256)).unwrap().into_dyn();
    let a1:Array<f32, IxDyn> = concatenate![Axis(1), a1, noise_arr];
    let b1:Array<f32, Ix4> = frames.slice(s![..,1,..,..,..]).to_owned();
    let b1 = b1.into_dyn();
    let c1 = (*block1.run::<'_, '_, '_, f32, f32, IxDyn>(vec![a1, b1]).unwrap()[0]).to_owned();

    let a2 : Array<f32, Ix5> = frames.slice(s![..,2..5,..,..,..]).to_owned();
    let a2 = a2.into_shape((1,9,256,256)).unwrap().into_dyn();
    let a2:Array<f32, IxDyn> = concatenate![Axis(1), a2, noise_arr];
    let b2:Array<f32, Ix4> = frames.slice(s![..,1,..,..,..]).to_owned();
    let b2 = b2.into_dyn();
    let c2 = (*block1.run::<'_, '_, '_, f32, f32, IxDyn>(vec![a2, b2]).unwrap()[0]).to_owned();

    let block2 = models.get_mut(&("block2_256".to_string())).unwrap();
    let d:Array<f32, IxDyn> = concatenate![Axis(1), c0, c1, c2, noise_arr];
    let e:Array<f32, IxDyn> = frames.slice(s![..,2,..,..,..]).to_owned().into_dyn();
    let result = (*block2.run::<'_, '_, '_, f32, f32, IxDyn>(vec![d, e]).unwrap()[0]).to_owned();

    let output: Array<f32, Ix4> = result.into_dimensionality::<Ix4>().unwrap();
    output
}

pub fn denoise_256(models: &mut HashMap<String, onnxruntime::session::Session<'_>>,
           arrs_all:Vec<(String, Array<f32, Ix4>, (u32,u32))>, noise_level:f32) -> Vec<(String, Array<f32, Ix4>, (u32,u32))>{
    let n_frames:usize = arrs_all.len();
    if n_frames >= 5 {
        let mut denoised_all: Vec<(String, Array<f32, Ix4>, (u32,u32))> = Vec::new();
        for i in 0..n_frames{
            // FIXME
            println!("Denoising {}", &arrs_all[i].0);
            let f0 = &arrs_all[if i<2 {0} else {i-2}].1;
            let f1 = &arrs_all[if i<1 {0} else {i-1}].1;
            let f2 = &arrs_all[i].1;
            let f3 = &arrs_all[if i>n_frames-2 {n_frames-1} else {i+1}].1;
            let f4 = &arrs_all[if i>n_frames-3 {n_frames-1} else {i+2}].1;
            let frames : Array<f32,Ix4> = ndarray::concatenate![Axis(0), f0.clone(), f1.clone(), f2.clone(), f3.clone(), f4.clone()];
            let frames : Array<f32,Ix5> = frames.insert_axis(Axis(0));
            let denoised = denoise_256_(models, frames, noise_level);
            denoised_all.push( (arrs_all[i].0.clone(), denoised, arrs_all[i].2) );
        }
        denoised_all
    }
    else{
        arrs_all
    }
}
