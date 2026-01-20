use core::time;
use std::{
  sync::{Arc,Mutex,mpsc::{Sender, channel}}, thread::{self, JoinHandle, sleep}
};

// 包装匿名函数类型
type Workfn = Box<dyn FnOnce()->() + Send + 'static>;
// 区分工作和停机消息
enum Msg {
  Work(Workfn),
  Down
}
// 使用Msg命名空间
use Msg::*;

// 主构造函数Concurrent
pub struct Concur {
  count: usize, // 线程数量
  sender: Sender<Msg>, // 异步发送器
  threads: Option<Vec<JoinHandle<()>>>, // 带有 原子指针 异步接收器 的线程 列表
  queued: Arc<Mutex<usize>>, // 已入队任务数量
}
impl Concur {
  // 初始化函数
  pub fn new(count: usize)-> Concur {
    let mut threads = Vec::with_capacity(count);
    let (sender,receiver) = channel();
    let receiver = Arc::new(Mutex::new(receiver));
    let queued = Arc::new(Mutex::new(0));
    for i in 0..count {
      let p_rec = Arc::clone(&receiver);
      let p_queued = Arc::clone(&queued);
      threads.push(thread::spawn(move || loop {
        let f: Msg = p_rec.lock().unwrap().recv().unwrap();
        match f {
          Work(f)=>{f();*p_queued.lock().unwrap() -= 1},
          Down=>{println!("{} down",i);break}
        };
      }));
    }
    Concur { count, sender, threads: Some (threads), queued }
  }
  // 实例的exec方法
  pub fn exec(&self,f:Workfn) {
    loop {
      let cur = *self.queued.lock().unwrap();
      if cur < self.count * 5 {
        break;
      }
      sleep(time::Duration::from_millis(100));
    }
    *self.queued.lock().unwrap() += 1;
    self.sender.send(Work(Box::new(f))).unwrap();
  }

  pub fn wait(&self) -> bool {
    let cur = *self.queued.lock().unwrap();
    cur >= self.count * 5
  }

  pub fn join(&mut self) {
    // 等待所有任务完成
    while *self.queued.lock().unwrap() > 0 {
      sleep(time::Duration::from_millis(100));
    }
  }
}

// Concur实例生命结束时会由rust运行drop()
impl Drop for Concur {
  fn drop(&mut self) {
    // 发送停机消息
    for _ in 0..self.count {
      self.sender.send(Down).unwrap();
    }
    // 等待所有线程运行完毕
    for thread in self.threads.take().unwrap() {
      thread.join().unwrap();
    }
  }
}