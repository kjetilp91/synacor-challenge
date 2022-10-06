use synacor_challenge::{Vm, MWord};

fn main() {
    let mut vm = Vm::default();
    vm.load_bytes_at(MWord::new(0), include_bytes!("../challenge.bin").as_ref());
    vm.run(&mut synacor_challenge::stdio());
}
