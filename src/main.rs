pub mod vector;

use std::{
    io::{self, Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use rand::{
    distributions::{Distribution, Standard},
    random,
};
use vector::{Vec2isize, Vec2usize};

fn main() {
    std::io::stdout().write_all(b"\x1b[2J").ok();

    // whether to update distances immediately or double-buffer them
    // double-buffering makes distance propagation visible
    const IMMEDIATE_DISTANCES: bool = false;

    // whether to update distances top-down left to right (false)
    // or starting from the root point and moving avay from it (true)
    // basically, no difference if IMMEDIATE_DISTANCES is false
    const DISTANCES_DIRECTIONAL: bool = false;

    // whether to allow crossing (and deleting) "walls" of the maze
    const NO_WALL_JUMPS: bool = false;

    const JUMP_HEIGHT: f32 = 1.0;
    const RELATIVE_JUMP_HEIGHT: bool = true;

    // the bigger the value, bigger change to continue moving in a straight line
    const CONTINUE_MOVING_CHANCE_DIV: usize = 5;

    // whether to create "particles" that will run away from the root
    const PARTICLES: bool = false;

    // less value - more particles
    const PARTICLE_CHANCE_DIV: usize = 100;

    const MOVE_DELAY: usize = 1;
    const TARGET_FPS: f32 = 60.0;

    let size = termsize::get().expect("terminal size");
    let size = Vec2usize::new(size.cols as usize, (size.rows as usize - 2) * 2);
    let sizei = size.convert(|v| v as isize);

    let mut maze = Field2d::new(size.x, size.y, || None::<Dir4>);
    let mut distances_a = Field2d::new(size.x, size.y, || 0);
    let mut root = size / 2;

    let mut pos = root;
    let mut max_dist = 0;
    let mut stack = vec![];

    'genloop: loop {
        let mut dirs = Dir4::ALL;

        for i in 0..4 {
            let other = random::<usize>() % 4;
            if i != other {
                dirs.swap(i, other);
            }
        }

        for d in dirs {
            let other_pos = pos.convert(|v| v as isize) + d.dir_isize();
            let other_pos = (other_pos + sizei) % sizei;
            let other_pos = other_pos.convert(|v| v as usize);

            if maze.get(other_pos).is_none() {
                *maze.get_mut(other_pos) = Some(d.inverted());
                stack.push(pos);
                pos = other_pos;

                *distances_a.get_mut(pos) = stack.len();
                max_dist = max_dist.max(stack.len());

                continue 'genloop;
            }
        }

        match stack.pop() {
            Some(p) => {
                pos = p;
            }
            None => break,
        }
    }

    let mut distances_b = distances_a.clone();
    let mut distances_src = &mut distances_a;
    let mut distances_dst = if !IMMEDIATE_DISTANCES {
        Some(&mut distances_b)
    } else {
        None
    };

    let mut trav_distances = vec![];

    let stdin_activity = Arc::new(AtomicBool::new(false));

    let stdin_activity_clone = stdin_activity.clone();
    thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        loop {
            match stdin.read(&mut [0]) {
                Ok(0) => continue,
                Err(_) | Ok(_) => {
                    stdin_activity_clone.store(true, Ordering::Relaxed);
                }
            }
        }
    });

    let mut move_countdown = 0;

    let mut prev_move_dir = Dir4::Up;
    let mut continuos_max_dist = max_dist;

    let mut fps_time_start = Instant::now();
    let mut fps_counter = 0;
    let mut fps = 0;

    

    let min_frame_time = Duration::from_secs_f32(1.0 / TARGET_FPS);

    loop {
        if stdin_activity.load(Ordering::Relaxed) {
            break;
        }

        let frame_time = Instant::now();

        // stdout_buf.clear();

        let update_start = Instant::now();

        if move_countdown == 0 {
            loop {
                let dir = if random::<usize>() % CONTINUE_MOVING_CHANCE_DIV == 0 {
                    let dir = random::<Dir4>();

                    if dir == prev_move_dir.inverted() {
                        random::<Dir4>()
                    } else {
                        dir
                    }
                } else {
                    prev_move_dir
                };
                let new_root = root.convert(|v| v as isize) + dir.dir_isize();
                let new_root = (new_root + sizei) % sizei;
                let new_root = new_root.convert(|v| v as usize);

                let jh = if RELATIVE_JUMP_HEIGHT {
                    (continuos_max_dist as f32 * JUMP_HEIGHT) as usize
                } else {
                    JUMP_HEIGHT as usize
                };

                if NO_WALL_JUMPS || *distances_src.get(new_root) > jh {
                    let new_root_dir = maze.get(new_root);
                    if new_root_dir.is_some_and(|d| d.inverted() != dir) {
                        let new_move_dir = random::<Dir4>();

                        if new_move_dir == prev_move_dir.inverted() {
                            prev_move_dir = random::<Dir4>();
                        } else {
                            prev_move_dir = new_move_dir;
                        }

                        continue;
                    }
                }

                if PARTICLES && random::<usize>() % PARTICLE_CHANCE_DIV == 0 {
                    *distances_src.get_mut(root) = *distances_src.get(new_root);
                } else {
                    *distances_src.get_mut(root) = 1;
                }

                *distances_src.get_mut(new_root) = 0;

                *maze.get_mut(root) = Some(dir);
                root = new_root;
                *maze.get_mut(root) = None;
                prev_move_dir = dir;

                move_countdown = MOVE_DELAY;
                break;
            }
        }

        move_countdown -= 1;

        let mut max_dist = 0;

        if DISTANCES_DIRECTIONAL {
            trav_distances.push(root);

            while let Some(pos) = trav_distances.pop() {
                let dir = *maze.get(pos);

                let dist = match dir {
                    None => 0,
                    Some(d) => {
                        (*distances_src.get(
                            ((pos.convert(|v| v as isize) + d.dir_isize() + sizei) % sizei)
                                .convert(|v| v as usize),
                        )) + 1
                    }
                };
                let dst = distances_dst.as_deref_mut().unwrap_or(distances_src);

                *dst.get_mut(pos) = dist;
                max_dist = max_dist.max(dist);

                for d in Dir4::ALL {
                    let other_pos = pos.convert(|v| v as isize) + d.dir_isize();
                    let other_pos = (other_pos + sizei) % sizei;
                    let other_pos = other_pos.convert(|v| v as usize);

                    let other_dir = *maze.get(other_pos);
                    if other_dir == Some(d.inverted()) {
                        trav_distances.push(other_pos);
                    }
                }
            }
        } else {
            for y in 0..size.y {
                for x in 0..size.x {
                    let pos = Vec2usize::new(x, y);
                    let dir = *maze.get(pos);

                    let dist = match dir {
                        None => 0,
                        Some(d) => {
                            (*distances_src.get(
                                ((pos.convert(|v| v as isize) + d.dir_isize() + sizei) % sizei)
                                    .convert(|v| v as usize),
                            )) + 1
                        }
                    };

                    let dst = distances_dst.as_deref_mut().unwrap_or(distances_src);
                    *dst.get_mut(pos) = dist;
                    max_dist = max_dist.max(dist);
                }
            }
        }

        if let Some(dst) = &mut distances_dst {
            std::mem::swap::<&mut _>(&mut distances_src, dst);
        }

        if continuos_max_dist > max_dist + 100 {
            continuos_max_dist -= 100;
        } else if continuos_max_dist + 100 < max_dist {
            continuos_max_dist += 100;
        } else if continuos_max_dist > max_dist {
            continuos_max_dist -= 1;
        } else if continuos_max_dist < max_dist {
            continuos_max_dist += 1;
        }

        let update_time = update_start.elapsed();

        let image = DistanceImage {
            max: continuos_max_dist,
            distances: distances_src,
        };

        let mut stdout = std::io::stdout().lock();

        let draw_start = Instant::now();

        draw(&image, &mut stdout);

        fps_counter += 1;

        let now = Instant::now();
        let secs = now
            .checked_duration_since(fps_time_start)
            .map(|d| d.as_secs_f32());
        if let Some(secs) = secs {
            if secs > 0.5 {
                let fpsf = fps_counter as f32 / secs;
                fps = fpsf.round() as u32;
                fps_time_start = now;
                fps_counter = 0;
                
            }
        }

        let draw_time = draw_start.elapsed();

        let update_ms = update_time.as_secs_f32() * 1000.0;
        let draw_ms = draw_time.as_secs_f32() * 1000.0;

        

        write!(stdout, "Press ENTER to stop | U: {update_ms:06.01}ms | D: {draw_ms:06.01}ms | max_dist: {continuos_max_dist} -> {max_dist} | fps: {fps}     ").ok();
        
        let frame_time = frame_time.elapsed();

        if let Some(diff) = min_frame_time.checked_sub(frame_time) {
            thread::sleep(diff);
        }
    }
}

struct DistanceImage<'a> {
    max: usize,
    distances: &'a Field2d<usize>,
}

impl Image2d for DistanceImage<'_> {
    fn width(&self) -> usize {
        self.distances.width()
    }

    fn height(&self) -> usize {
        self.distances.height()
    }

    fn get(&self, x: usize, y: usize) -> Rgb {
        let d = *self.distances.get([x, y]);

        if d == 0 {
            return [255, 0, 0];
        }

        let f = d as f32 / self.max as f32;
        // let f = f * 0.5 + 0.5;
        let b = (f * 255.0).clamp(0.0, 255.0).floor() as u8;

        // let b = d % 256;
        // let b = b.min(255) as u8;

        // let b = 255 - b;

        [b; 3]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir4 {
    Up,
    Right,
    Down,
    Left,
}

impl Dir4 {
    pub const ALL: [Self; 4] = [Self::Up, Self::Right, Self::Down, Self::Left];

    pub fn dir_isize(self) -> Vec2isize {
        match self {
            Self::Up => [0, -1],
            Self::Right => [1, 0],
            Self::Down => [0, 1],
            Self::Left => [-1, 0],
        }
        .into()
    }

    pub fn inverted(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Right => Self::Left,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
        }
    }
}

impl Distribution<Dir4> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Dir4 {
        match <Self as Distribution<u8>>::sample(self, rng) % 4 {
            0 => Dir4::Up,
            1 => Dir4::Right,
            2 => Dir4::Down,
            3 => Dir4::Left,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
struct Field2d<T> {
    width: usize,
    height: usize,
    data: Box<[T]>,
}

impl<T> Field2d<T> {
    fn new(width: usize, height: usize, mut maker: impl FnMut() -> T) -> Self {
        let mut vec = Vec::with_capacity(width * height);
        for _ in 0..width * height {
            vec.push(maker());
        }
        Self {
            width,
            height,
            data: vec.into(),
        }
    }

    pub fn get(&self, p: impl Into<Vec2usize>) -> &T {
        let p = p.into();
        &self.data[p.y * self.width + p.x]
    }

    pub fn get_mut(&mut self, p: impl Into<Vec2usize>) -> &mut T {
        let p = p.into();
        &mut self.data[p.y * self.width + p.x]
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }
}

type Rgb = [u8; 3];

trait Image2d {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn get(&self, x: usize, y: usize) -> Rgb;
}

struct RandomImage {
    width: usize,
    height: usize,
}

impl Image2d for RandomImage {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn get(&self, _x: usize, _y: usize) -> Rgb {
        random()
    }
}

fn draw<W: io::Write>(image: &dyn Image2d, w: &mut W) {
    w.write_all(b"\x1b[1;1H\x1b[?25l").ok();

    for y in 0..((image.height() + 1) / 2) {
        for x in 0..image.width() {
            let pxa = image.get(x, y * 2);
            let pxb = ((y * 2) + 1 < image.height()).then(|| image.get(x, (y * 2) + 1));

            w.write_fmt(format_args!("\x1b[38;2;{};{};{}m", pxa[0], pxa[1], pxa[2])).ok();

            match pxb {
                Some(c) => {
                    w.write_fmt(format_args!("\x1b[48;2;{};{};{}m", c[0], c[1], c[2])).ok();
                }
                None => {
                    w.write_all(b"\x1b[49m").ok();
                }
            }

            // '▀';
            w.write_all("▀".as_bytes()).ok();
        }

        w.write_all(b"\x1b[m\n").ok();
    }

    w.write_all(b"\x1b[?25h").ok();
}