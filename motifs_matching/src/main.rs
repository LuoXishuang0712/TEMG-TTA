mod concur;

use core::time;
use std::rc::Rc;
use getargs::{Opt, Options};
use once_cell::sync::Lazy;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::process::exit;
use std::sync::Mutex;
use std::sync::{mpsc, Arc};

use concur::Concur;

static MOTIFS_MODES: Lazy<HashMap<((u32, u32), (u32, u32), (u32, u32)), (u32, u32)>> = Lazy::new(|| HashMap::from([
    (((0, 2), (1, 2), (0, 2)), (0, 0)),
    (((0, 2), (1, 2), (2, 0)), (0, 1)),
    (((0, 2), (1, 2), (0, 1)), (0, 2)),
    (((0, 2), (1, 2), (1, 0)), (0, 3)),
    (((0, 2), (1, 2), (2, 1)), (0, 4)),
    (((0, 2), (1, 2), (1, 2)), (0, 5)),
    (((0, 2), (2, 1), (0, 2)), (1, 0)),
    (((0, 2), (2, 1), (2, 0)), (1, 1)),
    (((0, 2), (2, 1), (0, 1)), (1, 2)),
    (((0, 2), (2, 1), (1, 0)), (1, 3)),
    (((0, 2), (2, 1), (2, 1)), (1, 4)),
    (((0, 2), (2, 1), (1, 2)), (1, 5)),
    (((0, 2), (1, 0), (0, 2)), (2, 0)),
    (((0, 2), (1, 0), (2, 0)), (2, 1)),
    (((0, 2), (1, 0), (0, 1)), (2, 2)),
    (((0, 2), (1, 0), (1, 0)), (2, 3)),
    (((0, 2), (1, 0), (2, 1)), (2, 4)),
    (((0, 2), (1, 0), (1, 2)), (2, 5)),
    (((0, 2), (0, 1), (0, 2)), (3, 0)),
    (((0, 2), (0, 1), (2, 0)), (3, 1)),
    (((0, 2), (0, 1), (0, 1)), (3, 2)),
    (((0, 2), (0, 1), (1, 0)), (3, 3)),
    (((0, 2), (0, 1), (2, 1)), (3, 4)),
    (((0, 2), (0, 1), (1, 2)), (3, 5)),
    (((0, 2), (2, 0), (0, 2)), (4, 0)),
    (((0, 2), (2, 0), (2, 0)), (4, 1)),
    (((0, 2), (2, 0), (0, 1)), (4, 2)),
    (((0, 2), (2, 0), (1, 0)), (4, 3)),
    (((0, 2), (2, 0), (2, 1)), (4, 4)),
    (((0, 2), (2, 0), (1, 2)), (4, 5)),
    (((0, 2), (0, 2), (0, 2)), (5, 0)),
    (((0, 2), (0, 2), (2, 0)), (5, 1)),
    (((0, 2), (0, 2), (0, 1)), (5, 2)),
    (((0, 2), (0, 2), (1, 0)), (5, 3)),
    (((0, 2), (0, 2), (2, 1)), (5, 4)),
    (((0, 2), (0, 2), (1, 2)), (5, 5)),
]));
// motifs_inner_id: motif_id
// with in motifs, we consider the from and the to node of the first edge in time order as 0 and 2

#[derive(Debug)]
struct Args {
    input_file: Option<String>,
    output_file: Option<String>,
    time_range: u64,
    threads: usize,
    max_queue_len: usize,
    max_edges: Option<usize>,
    aggregration_window: u64,
    offload_disk_freq: usize,
    resume_file: Option<String>,
    output_timestamp: bool,
}

impl Args {
    pub fn new() -> Self {
        Self {
            input_file: None, 
            output_file: None, 
            time_range: 3_600, 
            threads: 16, 
            max_queue_len: 100_000, 
            max_edges: None, 
            aggregration_window: 0, 
            offload_disk_freq: 4,
            resume_file: None,
            output_timestamp: false,
        }
    }

    pub fn check_args(self: &Args) -> Result<(), String> {
        if self.input_file.is_none() || self.output_file.is_none() {
            return Err("input file is required".to_string());
        }
        if self.offload_disk_freq == 0 {
            return Err("offload_disk_freq must be greater than 0".to_string());
        }
        if self.offload_disk_freq > self.threads {
            eprintln!("a large offload_disk_freq will probably cause dead lock, please consider decrease it");
        }
        Ok(())
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut opts = Options::new(args.iter().map(String::as_str));
    let mut args = Args::new();

    while let Some(opt) = opts.next_opt().expect("opts") {
        match opt {
            Opt::Short('h') | Opt::Long("help") => {
                eprintln!(r"
Usage: This is a motifs matching algorithm for dynamic graphs implmentation in Rust.

Options:
    -h, --help  Show this help
    -i, --input <file>  The file to read from.
    -o, --output-file=<file>  Path to file
    -t, --time-range=<n>  The max time range one single window can capture.
    --threads=<n>  The number of threads to run in parallel. metion not all threads will be used as calculating, one of them will be used to handle i/o.
    -q, --max_queue_len=<n>  The number of motifs can be cached in memory, a larger number can increase the parallelism when came into a dense window, if you got into memory overflow, decrease it would be help.
    -e, --max_edges=<n>  The max edges one single time window will capture.
    --aggregration_window=<n>  The time range one direction edge will be aggregated.
    --offload_disk_freq=<n>  How many steps between once disk write, be caution a large value will cause dead lock.
    --resume_file=<file>  The file to resume from.
    --output_timestamp  Whether to output timestamp in the output file.

More info:
    # About the max_queue_len
        We cache motifs as bytes, which in format <motifs>\x20<node_id_1>\x20<node_id_2>\x20<node_id_3>\x21, and according to your graphs' nodes, the max node id will change.
        Under the circumstance which max node_id < 10_000, we can get a single motifs costs about 2(motifs) + 3 * 4(node_ids) + 4(seps) = 18 bytes.
        And alone with the Vec<> costs which is usually 24 bytes, we can got 42 totally.
        So, for a default 100_000 queue_len, we need 4_200_000 bytes aka 4.2MiB to queue motifs before it wrote into disks.
                ");
                exit(1)
            },
            Opt::Short('o') | Opt::Long("output_file") => args.output_file = Some(opts.value().unwrap().to_string()),
            Opt::Short('i') | Opt::Long("input_file") => args.input_file = Some(opts.value().unwrap().to_string()),
            Opt::Short('t') | Opt::Long("time_range") => args.time_range = opts.value().unwrap().to_string().parse::<u64>().unwrap(),
            Opt::Long("threads") => args.threads = opts.value().unwrap().to_string().parse::<usize>().unwrap(),
            Opt::Short('q') | Opt::Long("max_queue_len") => args.max_queue_len = opts.value().unwrap().to_string().parse::<usize>().unwrap(),
            Opt::Short('e') | Opt::Long("max_edges") => args.max_edges = Some(opts.value().unwrap().to_string().parse::<usize>().unwrap()),
            Opt::Long("aggregration_window") => args.aggregration_window = opts.value().unwrap().to_string().parse::<u64>().unwrap(),
            Opt::Long("offload_disk_freq") => args.offload_disk_freq = opts.value().unwrap().to_string().parse::<usize>().unwrap(),
            Opt::Long("resume_file") => args.resume_file = Some(opts.value().unwrap().to_string()),
            Opt::Long("output_timestamp") => args.output_timestamp = true,
            _ => {
                eprintln!("Unknown option {}, use --help for more information", opt.to_string());
                exit(1)
            }
        }
    }
    args.check_args().unwrap();
    let pid = std::process::id();
    println!("current pid: {}", pid);

    let mut resume_pos = if 
        let Some(resume_file) = args.resume_file.clone() && 
        let Ok(check_res) = fs::exists(&resume_file) && check_res && 
        let Ok(file_content) = fs::read_to_string(resume_file) &&
        let Some(line) = file_content.lines().next() &&
        let Ok(res) = str::parse::<usize>(&line.trim())
    {
        res
    } else {
        0_usize
    };

    let file = File::open(args.input_file.unwrap()).expect("Failed to open file");
    let output_file = Arc::new(Mutex::new(File::options().append(true).create(true).open(args.output_file.unwrap()).expect("Failed to create output file")));
    let resume_file = if let Some(resume_file) = args.resume_file.clone() {
        let mut f = File::create(resume_file).expect("Failed to create resume file");
        writeln!(f, "{}", resume_pos).expect("Failed to write resume file");
        Some(Rc::new(f))
    } else {
        None
    };
    let time_range: u64 = args.time_range;
    let edges = aggregration(read_file(file), args.aggregration_window);

    println!("read {} edges", edges.len());

    let mut thread_pool = Concur::new(args.threads);
    let mut dynamic_graph = DynamicGraph::new(edges, time_range);
    let (tx, rx) = mpsc::sync_channel(args.max_queue_len);
    // for each item, we have Vec<u8> which lens about (2 + 1 + 4 + 1 + 4 + 1 + 4 + 1) = 18, with vec its self costs 24, we need about 42 bytes to hold each
    // thus, we need about 42 * 1_000_000 = 42_000_000 bytes to hold 1_000_000 motifs, which is about 42 MB
    let rx_locked = Arc::new(Mutex::new(rx));

    let mut step = 0;
    loop {
        let r = DynamicGraph::next_step(&mut dynamic_graph);
        if r.is_err() {
            println!("{}", r.err().unwrap());
            break;
        }
        step += 1;
        if resume_pos != 0 && step < resume_pos {
            continue;
        }
        if resume_pos != 0 {
            println!("resume_pos: {} reached step: {}", resume_pos, step);
            resume_pos = 0;
        }
        let time_window = DynamicGraph::to_time_window(&dynamic_graph);
        let tx_cloned = tx.clone();
        thread_pool.exec(Box::new(move || {
            let motifs = worker(time_window, args.max_edges).unwrap();
            for motif in motifs {
                let bytes = motif.to_bytes(args.output_timestamp);
                tx_cloned.send(bytes).unwrap();
            }
        }));
        while step % args.offload_disk_freq == 0 {
            if let Some(resume_file) = resume_file.as_ref() {
                resume_file.as_ref().seek(SeekFrom::Start(0)).unwrap();
                writeln!(resume_file.as_ref(), "{}", step).expect("Failed to write resume file");
            }
            if let Err(_) = rx_locked.try_lock() {
                break;
            }
            let rx_cloned = rx_locked.clone();
            let output_file_cloned = output_file.clone();
            thread_pool.exec(Box::new(move || {
                let local_rx = rx_cloned.lock().unwrap();
                let mut buf: Vec<u8> = Vec::new();
                loop {
                    let recv_res = local_rx.recv_timeout(time::Duration::from_millis(10));
                    if recv_res.is_err() {
                        break;
                    }
                    let motifs = recv_res.unwrap();
                    buf.extend(motifs);
                    if buf.len() >= 1_000_000 {
                        output_file_cloned.lock().unwrap().write_all(&buf).unwrap();
                        buf.clear();
                    }
                }
                if !buf.is_empty() {
                    output_file_cloned.lock().unwrap().write_all(&buf).unwrap();
                }
            }));
            break;
        }
        if step % 10_000 == 0 {
            println!("step {} done", step);
        }
    }

    thread_pool.join();
    drop(tx);

    let rx_unlock = rx_locked.lock().unwrap();
    let mut locked_output_file = output_file.lock().unwrap();
    let mut buf: Vec<u8> = Vec::new();
    while let Ok(motifs_msg) = rx_unlock.recv() {
        buf.extend(motifs_msg);
        if buf.len() >= 1_000_000 {
            locked_output_file.write_all(&buf).unwrap();
            buf.clear();
        }
    }
    if !buf.is_empty() {
        locked_output_file.write_all(&buf).unwrap();
    }
}

struct MotifsResult {
    node_group: (u32, u32, u32),
    motifs_id: (u32, u32),
    timestamp: u32
}

impl MotifsResult {
    fn new(node_group: (u32, u32, u32), motifs_id: (u32, u32), timestamp: u32) -> Self {
        Self {
            node_group,
            motifs_id,
            timestamp
        }
    }

    fn to_bytes(&self, output_timestamp: bool) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(40);  // 5 + 5 + 5 + 2 + 10 + 7(sep) = 36
        append_u32_as_bytes(&mut buf, self.node_group.0);
        buf.push(b' ');
        append_u32_as_bytes(&mut buf, self.node_group.1);
        buf.push(b' ');
        append_u32_as_bytes(&mut buf, self.node_group.2);
        buf.push(b' ');
        append_u32_as_bytes(&mut buf, flat_motifs_id(self.motifs_id));
        if output_timestamp {
            buf.push(b' ');
            append_u32_as_bytes(&mut buf, self.timestamp);
        }
        buf.push(b'!');
        buf
    }
}

fn worker(time_window: TimeWindow, max_edges: Option<usize>) -> Result<Vec<MotifsResult>, String> {
    let motifs = time_window.match_window(max_edges)?;
    Ok(motifs)
}

fn flat_motifs_id(motifs_id: (u32, u32)) -> u32 {
    motifs_id.0 * 6 + motifs_id.1
}

// fn fmt_output_bytes(motif_res_item: ((u32, u32, u32), (u32, u32))) -> Vec<u8> {
//     // 元素用/x20(空格)分隔，条目用/x21(感叹号)分隔
//     let flat_id = flat_motifs_id(motif_res_item.1);
//     let (node0, node1, node2) = motif_res_item.0;
    
//     // 预先分配足够的容量以避免多次重新分配
//     // 假设每个u32最多需要10个字节(包括分隔符)
//     let mut result = Vec::with_capacity(40);
    
//     // 转换flat_id并添加分隔符
//     append_u32_as_bytes(&mut result, flat_id);
//     result.push(b' ');  // /x20
    
//     // 转换node0并添加分隔符
//     append_u32_as_bytes(&mut result, node0);
//     result.push(b' ');
    
//     // 转换node1并添加分隔符
//     append_u32_as_bytes(&mut result, node1);
//     result.push(b' ');
    
//     // 转换node2并添加右括号
//     append_u32_as_bytes(&mut result, node2);
    
//     // 添加条目分隔符/x21(感叹号)
//     result.push(b'!');  // /x21
    
//     result
// }

// 辅助函数：将u32转换为字节并附加到Vec<u8>中
fn append_u32_as_bytes(buffer: &mut Vec<u8>, value: u32) {
    if value == 0 {
        buffer.push(b'0');
        return;
    }
    
    let mut digits = [0u8; 10];  // u32最多10位数字
    let mut i = 9;
    let mut num = value;
    
    while num > 0 && i > 0 {
        digits[i] = b'0' + (num % 10) as u8;
        num /= 10;
        i -= 1;
    }
    
    // 跳过前导零，从第一个非零数字开始复制
    for j in (i+1)..10 {
        if digits[j] != 0 || j == 9 {
            buffer.push(digits[j]);
        }
    }
}

fn read_file(file: File) -> Vec<Arc<Edge>> {
    let reader = BufReader::new(file);
    let mut edges = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let mut parts = line.split_whitespace();
        let from = parts.next().expect("Failed to get row 'from'").parse().expect("Failed to parse from as u32");
        let to = parts.next().expect("Failed to get row 'to'").parse().expect("Failed to parse to as u32");
        let timestamp = parts.next().expect("Failed to get row 'timestamp'").parse().expect("Failed to parse timestamp as u64");
        if from == to {
            continue;  // ignore self-loop
        }
        edges.push(Arc::new(Edge::new(from, to, timestamp)));
    }
    edges.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    edges
}

fn aggregration(edges: Vec<Arc<Edge>>, aggregration_window: u64) -> Vec<Arc<Edge>> {
    if aggregration_window == 0 {
        return edges;
    }
    let mut result = Vec::new();
    let mut window_conn_cache: HashMap<(u32, u32), VecDeque<Arc<Edge>>> = HashMap::new();
    let mut window_cache: VecDeque<Arc<Edge>> = VecDeque::new();
    for edge in edges {
        let key = (edge.from, edge.to);
        let cur_ts = edge.timestamp;
        let not_exists = window_conn_cache.get(&key).is_none();
        window_cache.push_back(edge.clone());
        window_conn_cache.entry(key).or_insert(VecDeque::new()).push_back(edge.clone());

        while let Some(first_edge) = window_cache.front() && cur_ts - first_edge.timestamp > aggregration_window {
            let pop_key = (first_edge.from, first_edge.to);
            window_conn_cache.get_mut(&pop_key).unwrap().pop_front();
            if window_conn_cache.get(&pop_key).unwrap().len() == 0 {
                window_conn_cache.remove(&pop_key);
            }
            window_cache.pop_front();
        }

        if not_exists {
            result.push(edge.clone());
        }
    }

    result
}

struct TimeWindow {
    edges: VecDeque<Arc<Edge>>,
}

impl TimeWindow {
    fn new(edges: VecDeque<Arc<Edge>>) -> Self {
        Self {
            edges,
        }
    }

    fn match_window(&self, max_edges: Option<usize>) -> Result<Vec<MotifsResult>, String> {
        // return indicates: ((node_0_real_id, node_1_..., node_2_...), (*motif_id))
        // in specified window, we only consider the motifs that have at least one edge in the window, as the others will be matched in previous windows
        let last_edge = self.edges.back().expect("Failed to get last edge").clone();
        let mut result: Vec<MotifsResult> = Vec::new();

        let skip_count: usize = if let Some(max_edges) = max_edges && let edges_count = self.edges.len() && edges_count > max_edges {
            edges_count - max_edges
        } else {
            0
        };

        let root_from_node = last_edge.from;
        let root_to_node = last_edge.to;
        let root_timestamp = last_edge.timestamp as u32;

        for edge in self.edges.iter().skip(skip_count) {  // the first edge in motifs
            if Arc::ptr_eq(edge, &last_edge) || edge.timestamp >= last_edge.timestamp {
                continue;
            }
            let from_node = edge.from;
            let to_node = edge.to;
            
            if from_node != root_from_node && from_node != root_to_node && to_node != root_from_node && to_node != root_to_node {
                continue;
            }

            let node_0: u32 = from_node;
            let node_2: u32 = to_node;

            if (root_from_node == from_node && root_to_node == to_node) || (root_from_node == to_node && root_to_node == from_node) {
                let node_map = HashMap::from([
                    (node_0, 0),
                    (node_2, 2),
                ]);
                for edge_2 in self.edges.iter() {
                    if Arc::ptr_eq(edge_2, &last_edge) || edge_2.timestamp <= edge.timestamp {
                        continue;
                    }
                    let from_node_2 = edge_2.from;
                    let to_node_2 = edge_2.to;
                    if !node_map.contains_key(&from_node_2) && !node_map.contains_key(&to_node_2) {
                        continue;
                    }
                    if node_map.contains_key(&from_node_2) && node_map.contains_key(&to_node_2) {
                        let motifs_id = MOTIFS_MODES.get(&(
                            (0, 2),
                            (node_map[&from_node_2], node_map[&to_node_2]),
                            (node_map[&root_from_node], node_map[&root_to_node]),
                        ));

                        result.push(MotifsResult::new((node_0, 0, node_2), motifs_id.expect("Failed to get motifs id").clone(), root_timestamp));
                        continue;
                    }
                    let node_1 = if node_map.contains_key(&from_node_2) {
                        to_node_2
                    } else {
                        from_node_2
                    };
                    let mut node_map_cpy = node_map.clone();
                    node_map_cpy.insert(node_1, 1);
                    let k = &(
                        (0, 2),
                        (node_map_cpy[&from_node_2], node_map_cpy[&to_node_2]),
                        (node_map_cpy[&root_from_node], node_map_cpy[&root_to_node]),
                    );
                    // println!("{:?}", k);
                    let motifs_id = MOTIFS_MODES.get(k);

                    result.push(MotifsResult::new((node_0, node_1, node_2), motifs_id.expect("Failed to get motifs id").clone(), root_timestamp));
                }

                continue;
            }

            let node_1 = if root_from_node == from_node || root_from_node == to_node {
                root_to_node
            } else {
                root_from_node
            };
            let node_map = HashMap::from([
                (node_0, 0),
                (node_1, 1),
                (node_2, 2),
            ]);
            for edge_2 in self.edges.iter() {
                if Arc::ptr_eq(edge_2, &last_edge) || edge_2.timestamp <= edge.timestamp {
                    continue;
                }
                let from_node_2 = edge_2.from;
                let to_node_2 = edge_2.to;
                if !node_map.contains_key(&from_node_2) || !node_map.contains_key(&to_node_2) {
                    continue;
                }
                let k = &(
                    (0, 2),
                    (node_map[&from_node_2], node_map[&to_node_2]),
                    (node_map[&root_from_node], node_map[&root_to_node]),
                );
                // println!("{:?}", k);
                let motifs_id = MOTIFS_MODES.get(k);

                result.push(MotifsResult::new((node_0, node_1, node_2), motifs_id.expect("Failed to get motifs id").clone(), root_timestamp));
            }
        }

        Ok(result)
    }
}

#[derive(Debug)]
struct DynamicGraph {
    all_edges: Vec<Arc<Edge>>,
    time_range: u64,
    current_head_index: u32,
    current_tail_index: u32,
    current_time_window_edges: VecDeque<Arc<Edge>>,
    current_time_window_nodes_map: HashMap<u32, Mutex<Node>>,
}

impl DynamicGraph {
    fn new(all_edges: Vec<Arc<Edge>>, time_range: u64) -> Self {
        Self {
            all_edges,
            time_range,
            current_head_index: 0,
            current_tail_index: 0,
            current_time_window_edges: VecDeque::new(),
            current_time_window_nodes_map: HashMap::new(),
        }
    }

    fn next_step(&mut self) -> Result<(), String> {
        if self.current_tail_index >= self.all_edges.len() as u32 {
            return Err("No more edges".to_string());
        }

        self.current_tail_index += 1;
        let to_append_edge = self.all_edges[self.current_tail_index as usize - 1].clone();
        self.current_time_window_edges.push_back(to_append_edge.clone());
        
        let to_pop_edges: Vec<Arc<Edge>> = self.current_time_window_edges
            .iter()
            .filter(|edge| edge.timestamp < to_append_edge.timestamp.checked_sub(self.time_range).unwrap_or(0))
            .cloned()
            .collect();
        self.current_head_index += to_pop_edges.len() as u32;
        for _ in 0..to_pop_edges.len() {
            self.current_time_window_edges.pop_front();
        }

        for edge in to_pop_edges {
            let from_node = self.current_time_window_nodes_map.get(&edge.from).expect("Failed to get from node from time window map");
            let to_node = self.current_time_window_nodes_map.get(&edge.to).expect("Failed to get to node from time window map");
            let from_degree = from_node.lock().expect("Failed to lock for from node").pop_edge(&edge);
            let to_degree = to_node.lock().expect("Failed to lock for to node").pop_edge(&edge);

            if from_degree == 0 {
                self.current_time_window_nodes_map.remove(&edge.from);
            }
            if to_degree == 0 {
                self.current_time_window_nodes_map.remove(&edge.to);
            }
        }

        {
            if !self.current_time_window_nodes_map.contains_key(&to_append_edge.from) {
                self.current_time_window_nodes_map.insert(to_append_edge.from, Mutex::new(Node::new(to_append_edge.from)));
            }
            if !self.current_time_window_nodes_map.contains_key(&to_append_edge.to) {
                self.current_time_window_nodes_map.insert(to_append_edge.to, Mutex::new(Node::new(to_append_edge.to)));
            }
            
            let from_node = self.current_time_window_nodes_map.get(&to_append_edge.from).expect("Failed to get from node from time window map");
            let to_node = self.current_time_window_nodes_map.get(&to_append_edge.to).expect("Failed to get to node from time window map");
            from_node.lock().expect("Failed to lock for from node").push_edge(&to_append_edge);
            to_node.lock().expect("Failed to lock for to node").push_edge(&to_append_edge);
        }

        Ok(())
    }

    fn full_window(&self) -> bool {
        if self.current_head_index != 0 {
            return true;
        }
        if self.current_tail_index as usize + 1 >= self.all_edges.len() {
            return false;  // will never prepare a full window
        }

        let next_windows_edge = &self.all_edges[self.current_tail_index as usize + 1];
        let cur_head_edge = &self.all_edges[self.current_head_index as usize];
        
        next_windows_edge.timestamp - cur_head_edge.timestamp > self.time_range
    }

    fn to_time_window(&self) -> TimeWindow {
        TimeWindow::new(
            self.current_time_window_edges.clone(),
        )
    }
}

#[derive(Debug, Clone)]
struct Node {
    id: u32,
    in_edges: HashMap<u32, VecDeque<Arc<Edge>>>,  // nbr_id: edge(nbr -> cur)
    out_edges: HashMap<u32, VecDeque<Arc<Edge>>>,  // nbr_id: edge(cur -> nbr)
}

impl Node {
    fn new(id: u32) -> Self {
        Self {
            id,
            in_edges: HashMap::new(),
            out_edges: HashMap::new(),
        }
    }

    fn push_edge(&mut self, edge: &Arc<Edge>) {
        if edge.from == self.id {
            self.out_edges.entry(edge.to).or_default().push_back(edge.clone());
        } else if edge.to == self.id {
            self.in_edges.entry(edge.from).or_default().push_back(edge.clone());
        }
    }

    fn pop_edge(&mut self, edge: &Arc<Edge>) -> u32 {
        if edge.from == self.id {
            self.out_edges.get_mut(&edge.to).expect("Failed to get out edges for to node").pop_front();
            if self.out_edges.get(&edge.to).expect("Failed to get out edges for to node").is_empty() {
                self.out_edges.remove(&edge.to);
            }
        } else if edge.to == self.id {
            self.in_edges.get_mut(&edge.from).expect("Failed to get in edges for from node").pop_front();
            if self.in_edges.get(&edge.from).expect("Failed to get in edges for from node").is_empty() {
                self.in_edges.remove(&edge.from);
            }
        };

        self.in_edges.len() as u32 + self.out_edges.len() as u32  // not a accuracy degree, but a simple way to check if the node is still in the graph
    }
}

#[derive(Debug)]
struct Edge {
    from: u32,
    to: u32,
    timestamp: u64,
}

impl Edge {
    fn new(from: u32, to: u32, timestamp: u64) -> Self {
        Self { from, to, timestamp }
    }
}
