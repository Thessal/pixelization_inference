#![allow(non_snake_case)]

use std::{ops::Mul, ops::Sub, ops::Add};

//use image;
use std::collections::HashMap;
use onnxruntime::{
    ndarray::{Array, Ix2, Ix4, IxDyn, s},
};

pub fn grayscale(data: Array<f32, Ix4>)->Array<f32, Ix4>{
    // ITU-R 601-2 : L = R * 299/1000 + G * 587/1000 + B * 114/1000
    let R = data.slice(s![..,0..1,..,..]);
    let G = data.slice(s![..,1..2,..,..]);
    let B = data.slice(s![..,2..3,..,..]);
    let L : Array<f32, Ix4> = R.mul(0.299_f32) + G.mul(0.587_f32) + B.mul(0.114_f32);
    let L_arr = ndarray::concatenate(ndarray::Axis(1), &[L.view(), L.view(), L.view()]).unwrap();
    //let mut L_arr = Array::<f32, Ix4>::zeros(data.raw_dim());
    //L_arr.slice_mut(s![..,0..1,..,..]).assign(&L);
    //L_arr.slice_mut(s![..,1..2,..,..]).assign(&L);
    //L_arr.slice_mut(s![..,2..3,..,..]).assign(&L);
    L_arr
}

pub fn normalize(data: Array<f32, Ix4>)->Array<f32, Ix4>{
    let R = data.slice(s![..,0..1,..,..]);
    let G = data.slice(s![..,1..2,..,..]);
    let B = data.slice(s![..,2..3,..,..]);
    //let R_norm : Array<f32, Ix4> = (R.sub(R.mean().unwrap()))/(R.std(0.0_f32).max(0.1_f32));
    //let G_norm : Array<f32, Ix4> = (G.sub(G.mean().unwrap()))/(G.std(0.0_f32).max(0.1_f32));
    //let B_norm : Array<f32, Ix4> = (B.sub(B.mean().unwrap()))/(B.std(0.0_f32).max(0.1_f32));
    //let R_norm : Array<f32, Ix4> = ((R.div(255.0_f32))-0.5_f32) * 1.4_f32;
    //let G_norm : Array<f32, Ix4> = ((G.div(255.0_f32))-0.5_f32) * 1.4_f32; 
    //let B_norm : Array<f32, Ix4> = ((B.div(255.0_f32))-0.5_f32) * 1.4_f32;
    let R_norm : Array<f32, Ix4> = R.sub(0.5_f32) * 1.4_f32;
    let G_norm : Array<f32, Ix4> = G.sub(0.5_f32) * 1.4_f32; 
    let B_norm : Array<f32, Ix4> = B.sub(0.5_f32) * 1.4_f32;
    let N_arr = ndarray::concatenate(ndarray::Axis(1), &[R_norm.view(), G_norm.view(), B_norm.view()]).unwrap();
    N_arr
}

pub fn denormalize(data: Array<f32, Ix4>)->Array<f32, Ix4>{
    let R = data.slice(s![..,0..1,..,..]);
    let G = data.slice(s![..,1..2,..,..]);
    let B = data.slice(s![..,2..3,..,..]);
    let R_norm : Array<f32, Ix4> = R.add(1.0_f32).mul(0.5_f32).mapv(|v| v.max(0.).min(1.));
    let G_norm : Array<f32, Ix4> = G.add(1.0_f32).mul(0.5_f32).mapv(|v| v.max(0.).min(1.));
    let B_norm : Array<f32, Ix4> = B.add(1.0_f32).mul(0.5_f32).mapv(|v| v.max(0.).min(1.));
    let N_arr = ndarray::concatenate(ndarray::Axis(1), &[R_norm.view(), G_norm.view(), B_norm.view()]).unwrap();
    N_arr
}


fn g_a_RGBDec(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, 
              x:Array<f32, Ix4>, code:Array<f32, Ix2>) -> Array<f32, Ix4>{
    let mut x_ : Array<f32, IxDyn> = x.clone().into_dyn();
    let code_0 : Array<f32, IxDyn> = code.slice(s![..,..256]).to_owned().into_dyn();
    let code_1 : Array<f32, IxDyn> = code.slice(s![..,256*1..256*2]).to_owned().into_dyn();
    let code_2 : Array<f32, IxDyn> = code.slice(s![..,256*2..256*3]).to_owned().into_dyn();
    let code_3 : Array<f32, IxDyn> = code.slice(s![..,256*3..256*4]).to_owned().into_dyn();
    let code_4 : Array<f32, IxDyn> = code.slice(s![..,256*4..256*5]).to_owned().into_dyn();
    let code_5 : Array<f32, IxDyn> = code.slice(s![..,256*5..256*6]).to_owned().into_dyn();
    let code_6 : Array<f32, IxDyn> = code.slice(s![..,256*6..256*7]).to_owned().into_dyn();
    let code_7 : Array<f32, IxDyn> = code.slice(s![..,256*7..256*8]).to_owned().into_dyn();
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_1 = models.get_mut(&("g_a_RGBDec_mod_conv_1".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_1.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_0 ]).unwrap()[0]).to_owned();
        }
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_1 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    { // TODO : reduce redundancy
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_2 ]).unwrap()[0]).to_owned();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_3 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_4 ]).unwrap()[0]).to_owned();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_5 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let residual = x_.clone();
        {
            let g_a_RGBDec_mod_conv_2 = models.get_mut(&("g_a_RGBDec_mod_conv_2".to_string())).unwrap();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_6 ]).unwrap()[0]).to_owned();
            x_ = (*g_a_RGBDec_mod_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_, code_7 ]).unwrap()[0]).to_owned();
        }
        x_ = x_ + residual;
    }
    {
        let g_a_RGBDec_upsample_block1 = models.get_mut(&("g_a_RGBDec_upsample_block1".to_string())).unwrap();
        x_ = (*g_a_RGBDec_upsample_block1.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_conv_1 = models.get_mut(&("g_a_RGBDec_conv_1".to_string())).unwrap();
        x_ = (*g_a_RGBDec_conv_1.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_upsample_block2 = models.get_mut(&("g_a_RGBDec_upsample_block2".to_string())).unwrap();
        x_ = (*g_a_RGBDec_upsample_block2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_conv_2 = models.get_mut(&("g_a_RGBDec_conv_2".to_string())).unwrap();
        x_ = (*g_a_RGBDec_conv_2.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }
    {
        let g_a_RGBDec_conv_3 = models.get_mut(&("g_a_RGBDec_conv_3".to_string())).unwrap();
        x_ = (*g_a_RGBDec_conv_3.run::<'_, '_, '_, f32, f32, IxDyn>( vec![x_]).unwrap()[0]).to_owned();
    }

    x_.into_dimensionality::<Ix4>().unwrap()
}

fn g_a<'a>(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, 
       clipart:Array<f32, Ix4>, pixelart:Array<f32, Ix4>) -> Array<f32, Ix4>{
    let clipart_ = vec![clipart];
    let pixelart_ = vec![pixelart];
    let feature : Array<f32, Ix4>;
    let code : Array<f32, IxDyn>;
    let adain_params : Array<f32, Ix2>;
    {
        let g_a_RGBEnc = models.get_mut(&("g_a_RGBEnc".to_string())).unwrap();
        feature = (*g_a_RGBEnc.run( clipart_ ).unwrap()[0]).to_owned().into_dimensionality::<Ix4>().unwrap();
    }
    {
        let g_a_PBEnc = models.get_mut(&("g_a_PBEnc".to_string())).unwrap();
        code = (*g_a_PBEnc.run(pixelart_).unwrap()[0]).to_owned();
    }
    {
        let g_a_MLP = models.get_mut(&("g_a_MLP".to_string())).unwrap();
        adain_params = (*g_a_MLP.run(vec![code]).unwrap()[0]).to_owned().into_dimensionality::<Ix2>().unwrap();
    }
    let images : Array<f32, Ix4> = g_a_RGBDec(models, feature, adain_params).into_dimensionality::<Ix4>().unwrap();
    images
}

pub fn process_image(models: &mut HashMap<String, onnxruntime::session::Session<'_>>, 
                 data: Array<f32, Ix4>, reference: Array<f32, Ix4>) -> Array<f32, Ix4> {
    let mut images : Array<f32, IxDyn> = g_a(models, data, reference).into_dyn();
    {
        let alias_RGBEnc = models.get_mut(&("alias_RGBEnc".to_string())).unwrap();
        images = (*alias_RGBEnc.run( vec![images] ).unwrap()[0]).to_owned()
    }
    {
        let alias_RGBDec = models.get_mut(&("alias_RGBDec".to_string())).unwrap();
        images = (*alias_RGBDec.run( vec![images] ).unwrap()[0]).to_owned()
    }
    let output : Array<f32, Ix4> = images.into_dimensionality::<Ix4>().unwrap();
    output 
}

