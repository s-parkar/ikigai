use std::io::{Read, Write, BufReader, BufWriter};
use std::process::{Command, Stdio};
use std::fs::File;
use std::time::Instant;
use std::sync::Arc;
use rayon::prelude::*;

const IN_W: usize = 1920;
const IN_H: usize = 1080;
const IN_FRAME_BYTES: usize = IN_W * IN_H * 3;

const OUT_W: usize = 3200;
const OUT_H: usize = 1080;
const OUT_FRAME_BYTES: usize = OUT_W * OUT_H * 3;

#[derive(Clone)]
struct PixelMap {
    l_offset: i32,
    r_offset: i32,
    wl: u16,
    wr: u16,
}

fn process_stream(
    lhs: String, rhs: String, output: String,
    start_sec: f64, dur_sec: Option<f64>,
    maps: Arc<Vec<PixelMap>>, lut: Arc<Vec<u8>>,
    ffmpeg_bin: String
) {
    let temp_audio = format!("temp_audio_rust_{}.aac", std::process::id());
    let mut cmd_audio = Command::new(&ffmpeg_bin);
    cmd_audio.args(["-y", "-ss", &format!("{:.3}", start_sec)]);
    if let Some(dur) = dur_sec {
        cmd_audio.args(["-t", &format!("{:.3}", dur)]);
    }
    cmd_audio.args(["-i", &lhs, "-vn", "-c:a", "copy", &temp_audio]);
    let _ = cmd_audio.status();

    let mut cmd_dec_l = Command::new(&ffmpeg_bin);
    cmd_dec_l.args(["-hwaccel", "cuda", "-ss", &format!("{:.3}", start_sec)]);
    if let Some(dur) = dur_sec {
        cmd_dec_l.args(["-t", &format!("{:.3}", dur)]);
    }
    cmd_dec_l.args(["-i", &lhs, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]);
    cmd_dec_l.stdout(Stdio::piped()).stderr(Stdio::null());

    let mut cmd_dec_r = Command::new(&ffmpeg_bin);
    cmd_dec_r.args(["-hwaccel", "cuda", "-ss", &format!("{:.3}", start_sec)]);
    if let Some(dur) = dur_sec {
        cmd_dec_r.args(["-t", &format!("{:.3}", dur)]);
    }
    cmd_dec_r.args(["-i", &rhs, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]);
    cmd_dec_r.stdout(Stdio::piped()).stderr(Stdio::null());

    let mut cmd_enc = Command::new(&ffmpeg_bin);
    cmd_enc.args([
        "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", &format!("{}x{}", OUT_W, OUT_H),
        "-pix_fmt", "bgr24",
        "-r", "29.97",
        "-i", "pipe:0"
    ]);
    if std::path::Path::new(&temp_audio).exists() {
        cmd_enc.args(["-i", &temp_audio, "-map", "0:v", "-map", "1:a:0?", "-c:a", "aac"]);
    }
    cmd_enc.args([
        "-c:v", "h264_nvenc", "-preset", "p2", "-tune", "ll", "-gpu", "0",
        "-rc", "vbr", "-cq", "15", "-b:v", "65M", "-maxrate", "85M", "-bufsize", "100M",
        "-pix_fmt", "yuv420p",
        "-shortest",
        &output
    ]);
    cmd_enc.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());

    let mut proc_l = cmd_dec_l.spawn().expect("Failed to start LHS decoder");
    let mut proc_r = cmd_dec_r.spawn().expect("Failed to start RHS decoder");
    let mut proc_enc = cmd_enc.spawn().expect("Failed to start NVENC encoder");

    let mut reader_l = BufReader::with_capacity(16 * 1024 * 1024, proc_l.stdout.take().unwrap());
    let mut reader_r = BufReader::with_capacity(16 * 1024 * 1024, proc_r.stdout.take().unwrap());
    let mut writer_enc = BufWriter::with_capacity(16 * 1024 * 1024, proc_enc.stdin.take().unwrap());

    let mut buf_l = vec![0u8; IN_FRAME_BYTES];
    let mut buf_r = vec![0u8; IN_FRAME_BYTES];
    let mut buf_out = vec![0u8; OUT_FRAME_BYTES];

    let t0 = Instant::now();
    let mut count = 0usize;
    let total_est = if let Some(dur) = dur_sec { (dur * 29.97) as usize } else { 13088 };

    eprintln!("[Zentropy Rust Engine] Rayon Multi-Threaded SIMD Pipeline Active! Stitching {} frames...", total_est);

    loop {
        if let Err(_) = reader_l.read_exact(&mut buf_l) { break; }
        if let Err(_) = reader_r.read_exact(&mut buf_r) { break; }

        let l_slice = &buf_l[..];
        let r_slice = &buf_r[..];
        let maps_slice = &maps[..];
        let lut_slice = &lut[..];

        buf_out.par_chunks_mut(OUT_W * 3).enumerate().for_each(|(row, out_row)| {
            let row_offset = row * OUT_W;
            for col in 0..OUT_W {
                let p_idx = row_offset + col;
                let pm = &maps_slice[p_idx];
                let out_px = col * 3;

                let (b_l, g_l, r_l) = if pm.l_offset >= 0 {
                    let off = pm.l_offset as usize;
                    unsafe { (*l_slice.get_unchecked(off), *l_slice.get_unchecked(off+1), *l_slice.get_unchecked(off+2)) }
                } else { (0, 0, 0) };

                let (b_r, g_r, r_r) = if pm.r_offset >= 0 {
                    let off = pm.r_offset as usize;
                    let raw_b = unsafe { *r_slice.get_unchecked(off) } as usize >> 3;
                    let raw_g = unsafe { *r_slice.get_unchecked(off+1) } as usize >> 3;
                    let raw_r = unsafe { *r_slice.get_unchecked(off+2) } as usize >> 3;
                    let lut_off = (raw_b * 32 * 32 + raw_g * 32 + raw_r) * 3;
                    unsafe {
                        (*lut_slice.get_unchecked(lut_off), *lut_slice.get_unchecked(lut_off+1), *lut_slice.get_unchecked(lut_off+2))
                    }
                } else { (0, 0, 0) };

                if pm.wl > 0 && pm.wr > 0 {
                    out_row[out_px] = ((b_l as u32 * pm.wl as u32 + b_r as u32 * pm.wr as u32) >> 8) as u8;
                    out_row[out_px+1] = ((g_l as u32 * pm.wl as u32 + g_r as u32 * pm.wr as u32) >> 8) as u8;
                    out_row[out_px+2] = ((r_l as u32 * pm.wl as u32 + r_r as u32 * pm.wr as u32) >> 8) as u8;
                } else if pm.wl > 0 {
                    out_row[out_px] = b_l;
                    out_row[out_px+1] = g_l;
                    out_row[out_px+2] = r_l;
                } else if pm.wr > 0 {
                    out_row[out_px] = b_r;
                    out_row[out_px+1] = g_r;
                    out_row[out_px+2] = r_r;
                } else {
                    out_row[out_px] = 0;
                    out_row[out_px+1] = 0;
                    out_row[out_px+2] = 0;
                }
            }
        });

        if let Err(_) = writer_enc.write_all(&buf_out) { break; }
        count += 1;

        if count % 30 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let cur_fps = count as f64 / elapsed;
            let eta = if cur_fps > 0.0 && total_est > count { (total_est - count) as f64 / cur_fps } else { 0.0 };
            println!("PROGRESS:{}|{}|{:.1}|{:.1}|{:.1}", count, total_est, cur_fps, elapsed, eta);
        }
    }

    let _ = writer_enc.flush();
    drop(writer_enc);
    let _ = proc_enc.wait();
    let _ = proc_l.wait();
    let _ = proc_r.wait();

    if std::path::Path::new(&temp_audio).exists() {
        let _ = std::fs::remove_file(&temp_audio);
    }

    let total_elapsed = t0.elapsed().as_secs_f64();
    let final_fps = count as f64 / total_elapsed;
    eprintln!("[Zentropy Rust Engine] SUCCESS: Stitched {} frames in {:.2}s -> {:.1} FPS!", count, total_elapsed, final_fps);
    println!("PROGRESS:{}|{}|{:.1}|{:.1}|{:.1}", count, total_est, final_fps, total_elapsed, 0.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut lhs = "LHS.MOV".to_string();
    let mut rhs = "RHS.MOV".to_string();
    let mut output = "stitched_panorama_full.mp4".to_string();
    let mut map_file = "stitch_maps.bin".to_string();
    let mut ffmpeg_bin = r"C:\Users\yashs\ffmpeg-7.1-full_build-shared\bin\ffmpeg.exe".to_string();
    let mut start_sec = 0.0f64;
    let mut dur_sec: Option<f64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lhs" => { lhs = args[i+1].clone(); i += 2; }
            "--rhs" => { rhs = args[i+1].clone(); i += 2; }
            "--output" => { output = args[i+1].clone(); i += 2; }
            "--maps" => { map_file = args[i+1].clone(); i += 2; }
            "--ffmpeg" => { ffmpeg_bin = args[i+1].clone(); i += 2; }
            "--start" => {
                let parts: Vec<&str> = args[i+1].split(':').collect();
                if parts.len() == 3 {
                    let h: f64 = parts[0].parse().unwrap_or(0.0);
                    let m: f64 = parts[1].parse().unwrap_or(0.0);
                    let s: f64 = parts[2].parse().unwrap_or(0.0);
                    start_sec = h * 3600.0 + m * 60.0 + s;
                } else {
                    start_sec = args[i+1].parse().unwrap_or(0.0);
                }
                i += 2;
            }
            "--duration" => {
                let val_str = args[i+1].trim();
                if !val_str.is_empty() && val_str != "0" {
                    if let Ok(d) = val_str.parse::<f64>() {
                        dur_sec = Some(d);
                    }
                }
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    eprintln!("[Zentropy Rust Engine] Loading precomputed LUT and geometric maps from {}", map_file);

    let map_data = match File::open(&map_file) {
        Ok(mut f) => {
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            buf
        }
        Err(e) => {
            eprintln!("[Zentropy Rust Engine] Error opening map file {}: {}", map_file, e);
            return;
        }
    };

    let lut_size = 32 * 32 * 32 * 3;
    let lut = Arc::new(map_data[0..lut_size].to_vec());

    let map_entries = OUT_W * OUT_H;
    let mut pixel_maps: Vec<PixelMap> = Vec::with_capacity(map_entries);

    let mut ptr = lut_size;
    for _ in 0..map_entries {
        let l_off = i32::from_le_bytes(map_data[ptr..ptr+4].try_into().unwrap());
        let r_off = i32::from_le_bytes(map_data[ptr+4..ptr+8].try_into().unwrap());
        let wl = u16::from_le_bytes(map_data[ptr+8..ptr+10].try_into().unwrap());
        let wr = u16::from_le_bytes(map_data[ptr+10..ptr+12].try_into().unwrap());
        pixel_maps.push(PixelMap { l_offset: l_off, r_offset: r_off, wl, wr });
        ptr += 12;
    }
    let maps_arc = Arc::new(pixel_maps);

    process_stream(lhs, rhs, output, start_sec, dur_sec, maps_arc, lut, ffmpeg_bin);
}



