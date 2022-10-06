use std::io::{Write, BufRead};

pub trait Term {
    fn write(&mut self, c: u8) -> std::io::Result<()>;
    fn read_line(&mut self, dst: &mut Vec<u8>) -> std::io::Result<()>;
}

pub struct SimpleTerm<R: BufRead, W: Write> {
    stdin: R,
    stdout: W,
}

impl<R: BufRead, W: Write> Term for SimpleTerm<R, W> {
    fn write(&mut self, c: u8) -> std::io::Result<()> {
        self.stdout.write_all(&[c])
    }

    fn read_line(&mut self, dst: &mut Vec<u8>) -> std::io::Result<()> {
        self.stdin.read_until(b'\n', dst).map(|_| ())
    }
}

pub fn stdio() -> SimpleTerm<impl BufRead, impl Write> {
    SimpleTerm {
        stdin: std::io::BufReader::new(std::io::stdin()),
        stdout: std::io::stdout(),
    }
}

#[derive(Default)]
pub struct BufferTerm<'a> {
    stdin: Vec<&'a [u8]>,
    stdin_pos: usize,
    stdout: Vec<u8>,
}

impl<'a> BufferTerm<'a> {
    pub fn push_line(&mut self, data: &'a [u8]) {
        self.stdin.push(data);
    }
}

impl<'a> Term for BufferTerm<'a> {
    fn write(&mut self, c: u8) -> std::io::Result<()> {
        self.stdout.push(c);
        Ok(())
    }

    fn read_line(&mut self, dst: &mut Vec<u8>) -> std::io::Result<()> {
        if let Some(line) = self.stdin.get(self.stdin_pos) {
            dst.extend_from_slice(line);
            dst.push(b'\n');
        } else {
            todo!()
        }
        Ok(())
    }
}


mod mword {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct MWord(u16);

    impl MWord {
        pub(crate) const COUNT: u16 = 1 << 15;
        pub(crate) const MAX: u16 = Self::COUNT - 1;

        pub fn new(v: u16) -> Self {
            assert!(v <= Self::MAX);
            Self(v)
        }

        pub(crate) const fn as_addr(self) -> usize {
            self.0 as usize
        }

        pub(crate) const fn as_u32(self) -> u32 {
            self.0 as u32
        }

        pub(crate) const fn as_u16(self) -> u16 {
            self.0 as u16
        }

        pub(crate) const fn as_u8(self) -> u8 {
            self.0 as u8
        }

        pub(crate) fn run_op(self, other: Self, op: impl FnOnce(u32, u32) -> u32) -> Self {
            Self(op(u32::from(self.0), u32::from(other.0)) as u16 % Self::COUNT)
        }

        pub fn ret_inc_addr(&mut self) -> usize {
            let ret = self.as_addr();
            self.0 += 1;
            self.0 %= Self::COUNT;
            ret
        }
    }
}

pub use mword::MWord;

enum OpCode {
    /// halt: 0
    /// stop execution and terminate the program
    Halt,
    /// set: 1 a b
    /// set register `a` to the value of `b`
    Set(u8, MWord),
    /// push: 2 a
    /// push `a` onto the stack
    Push(MWord),
    /// pop: 3 a
    /// remove the top element from the stack and write it into `a`; empty stack = error
    Pop(Imm),
    /// eq: 4 a b c
    /// set `a` to 1 if `b` is equal to `c`; set it to 0 otherwise
    Eq(Imm3),
    /// gt: 5 a b c
    /// set `a` to 1 if `b` is greater than `c`; set it to 0 otherwise
    Gt(Imm3),
    /// jmp: 6 a
    /// jump to `a`
    Jmp(MWord),
    /// jt: 7 a b
    /// if `a` is nonzero, jump to `b`
    Jt(MWord, MWord),
    /// jf: 8 a b
    /// if `a` is zero, jump to `b`
    Jf(MWord, MWord),
    /// add: 9 a b c
    /// assign into `a` the sum of `b` and `c` (modulo 32768)
    Add(Imm3),
    /// mult: 10 a b c
    /// store into `a` the product of `b` and `c` (modulo 32768)
    Mult(Imm3),
    /// mod: 11 a b c
    /// store into `a` the remainder of `b` divided by `c`
    Mod(Imm3),
    /// and: 12 a b c
    /// stores into `a` the bitwise and of `b` and `c`
    And(Imm3),
    /// or: 13 a b c
    /// stores into `a` the bitwise or of `b` and `c`
    Or(Imm3),
    /// not: 14 a b
    /// stores 15-bit bitwise inverse of `b` in `a`
    Not(Imm, MWord),
    /// rmem: 15 a b
    /// read memory at address `b` and write it to `a`
    RMem(Imm, MWord),
    /// wmem: 16 a b
    /// write the value from `b` into memory at address `a`
    WMem(MWord, MWord),
    /// call: 17 a
    /// write the address of the next instruction to the stack and jump to `a`
    Call(MWord),
    /// ret: 18
    /// remove the top element from the stack and jump to it; empty stack = halt
    Ret,
    /// out: 19 a
    /// write the character represented by ascii code `a` to the terminal
    Out(u8),
    /// in: 20 a
    /// read a character from the terminal and write its ascii code to `a`; it can be assumed that
    /// once input starts, it will continue until a newline is encountered; this means that you can
    /// safely read whole lines from the keyboard and trust that they will be fully read
    In(Imm),
    /// noop: 21
    /// no operation
    Noop,
}

enum Imm3 {
    ToReg(u8, MWord, MWord),
    ToMem(MWord, MWord, MWord),
}

enum Imm {
    Value(MWord),
    Reg(u8),
}

pub struct Vm {
    registers: [MWord; 8],
    ip: MWord,
    memory: Box<[u16; MWord::COUNT as usize]>,
    stack: Vec<MWord>,
    cur_input: Vec<u8>,
    cur_input_pos: usize,
}

impl Default for Vm {
    fn default() -> Self {
        Vm {
            registers: [MWord::new(0); 8],
            ip: MWord::new(0),
            memory: Box::new([0; MWord::COUNT as usize]),
            stack: Vec::new(),
            cur_input: Vec::new(),
            cur_input_pos: 0,
        }
    }
}

impl Vm {
    pub fn load_bytes_at(&mut self, start_addr: MWord, data: &[u8]) {
        assert_eq!(data.len() % 2, 0);
        assert!(start_addr.as_addr() + (data.len() / 2) <= MWord::COUNT as usize);

        // ToDo: use array_chunks_exact when stable
        for (mem, inp) in self.memory[start_addr.as_addr()..].iter_mut().zip(data.chunks_exact(2)) {
            *mem = u16::from_le_bytes([inp[0], inp[1]]);
        }
    }

    pub fn load_words_at(&mut self, start_addr: MWord, data: &[u16]) {
        self.memory[start_addr.as_addr()..][..data.len()].copy_from_slice(data);
    }

    fn read_imm(&self, addr: &mut MWord) -> Imm {
        let value = self.memory[addr.ret_inc_addr()];
        match value {
            0..=MWord::MAX => Imm::Value(MWord::new(value)),
            MWord::COUNT..=32775 => Imm::Reg((value - MWord::COUNT) as u8),
            _ => panic!("Invalid value {value:?}"),
        }
    }

    fn read_imm_reg(&self, addr: &mut MWord) -> u8 {
        match self.read_imm(addr) {
            Imm::Value(v) => panic!("Invalid opcode, expected register got {v:?}"),
            Imm::Reg(reg) => reg,
        }
    }

    fn read_imm_value(&self, addr: &mut MWord) -> MWord {
        match self.read_imm(addr) {
            Imm::Value(v) => v,
            Imm::Reg(reg) => self.registers[usize::from(reg)],
        }
    }

    fn read_imm3(&self, addr: &mut MWord) -> Imm3 {
        let a = self.read_imm(addr);
        let b = self.read_imm_value(addr);
        let c = self.read_imm_value(addr);
        match a {
            Imm::Value(a) => Imm3::ToMem(a, b, c),
            Imm::Reg(reg) => Imm3::ToReg(reg, b, c),
        }
    }

    fn decode_at(&self, mut addr: MWord) -> (OpCode, MWord) {
        match self.memory[addr.ret_inc_addr()] {
            0 => (OpCode::Halt, addr),
            1 => (OpCode::Set(self.read_imm_reg(&mut addr), self.read_imm_value(&mut addr)), addr),
            2 => (OpCode::Push(self.read_imm_value(&mut addr)), addr),
            3 => (OpCode::Pop(self.read_imm(&mut addr)), addr),
            4 => (OpCode::Eq(self.read_imm3(&mut addr)), addr),
            5 => (OpCode::Gt(self.read_imm3(&mut addr)), addr),
            6 => (OpCode::Jmp(self.read_imm_value(&mut addr)), addr),
            7 => (OpCode::Jt(self.read_imm_value(&mut addr), self.read_imm_value(&mut addr)), addr),
            8 => (OpCode::Jf(self.read_imm_value(&mut addr), self.read_imm_value(&mut addr)), addr),
            9 => (OpCode::Add(self.read_imm3(&mut addr)), addr),
            10 => (OpCode::Mult(self.read_imm3(&mut addr)), addr),
            11 => (OpCode::Mod(self.read_imm3(&mut addr)), addr),
            12 => (OpCode::And(self.read_imm3(&mut addr)), addr),
            13 => (OpCode::Or(self.read_imm3(&mut addr)), addr),
            14 => (OpCode::Not(self.read_imm(&mut addr), self.read_imm_value(&mut addr)), addr),
            15 => (OpCode::RMem(self.read_imm(&mut addr), self.read_imm_value(&mut addr)), addr),
            16 => (OpCode::WMem(self.read_imm_value(&mut addr), self.read_imm_value(&mut addr)), addr),
            17 => (OpCode::Call(self.read_imm_value(&mut addr)), addr),
            18 => (OpCode::Ret, addr),
            19 => (OpCode::Out(self.read_imm_value(&mut addr).as_u8()), addr),
            20 => (OpCode::In(self.read_imm(&mut addr)), addr),
            21 => (OpCode::Noop, addr),
            unk => todo!("Unknown opcode {unk:?}"),
        }
    }

    fn decode(&mut self) -> OpCode {
        let (opcode, new_ip) = self.decode_at(self.ip);
        self.ip = new_ip;
        opcode
    }

    fn run_imm3_op(&mut self, imm: Imm3, op: impl FnOnce(u32, u32) -> u32) {
        match imm {
            Imm3::ToReg(reg, b, c) => self.registers[usize::from(reg)] = b.run_op(c, op),
            Imm3::ToMem(_, _, _) => todo!(),
        }
    }

    fn store_imm(&mut self, dst: Imm, value: MWord) {
        match dst {
            Imm::Value(mem) => todo!("{mem:?}"),
            Imm::Reg(reg) => self.registers[usize::from(reg)] = value,
        }
    }

    pub fn execute_one(&mut self, term: &mut impl Term) -> bool {
        match self.decode() {
            OpCode::Halt => return false,
            OpCode::Set(reg, value) => self.registers[usize::from(reg)] = value,
            OpCode::Push(value) => self.stack.push(value),
            OpCode::Pop(dst) => {
                let value = self.stack.pop().unwrap();
                self.store_imm(dst, value);
            }
            OpCode::Eq(imm) => self.run_imm3_op(imm, |b, c| (b == c).into()),
            OpCode::Gt(imm) => self.run_imm3_op(imm, |b, c| (b > c).into()),
            OpCode::Jmp(dst) => self.ip = dst,
            OpCode::Jt(a, b) => if a.as_u32() != 0 { self.ip = b },
            OpCode::Jf(a, b) => if a.as_u32() == 0 { self.ip = b },
            OpCode::Add(imm) => self.run_imm3_op(imm, |b, c| b + c),
            OpCode::Mult(imm) => self.run_imm3_op(imm, |b, c| b * c),
            OpCode::Mod(imm) => self.run_imm3_op(imm, |b, c| b % c),
            OpCode::And(imm) => self.run_imm3_op(imm, |b, c| b & c),
            OpCode::Or(imm) => self.run_imm3_op(imm, |b, c| b | c),
            OpCode::Not(dst, value) => self.store_imm(dst, value.run_op(value, |v, _| !v)),
            OpCode::RMem(dst, addr) => {
                let value = self.memory[addr.as_addr()];
                self.store_imm(dst, MWord::new(value));
            }
            OpCode::WMem(dst, value) => self.memory[dst.as_addr()] = value.as_u16(),
            OpCode::Call(dst) => {
                self.stack.push(self.ip);
                self.ip = dst;
            },
            OpCode::Ret => {
                if let Some(dst) = self.stack.pop() {
                    self.ip = dst;
                } else {
                    return false;
                }
            },
            OpCode::Out(c) => term.write(c).unwrap(),
            OpCode::In(dst) => {
                if self.cur_input.len() == self.cur_input_pos {
                    self.cur_input.clear();
                    self.cur_input_pos = 0;
                    term.read_line(&mut self.cur_input).unwrap();
                    assert_eq!(self.cur_input.last(), Some(&b'\n'));
                }
                let value = MWord::new(self.cur_input[self.cur_input_pos].into());
                self.cur_input_pos += 1;
                self.store_imm(dst, value);
            },
            OpCode::Noop => {}
        }
        true
    }

    pub fn run(&mut self, term: &mut impl Term) {
        loop {
            if !self.execute_one(term) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_program() {
        let mut term = BufferTerm::default();
        let mut vm = Vm::default();
        vm.load_words_at(MWord::new(0), &[9, 32768, 32769, 4, 19, 32768]);
        vm.run(&mut term);
        assert_eq!(vm.registers[0], MWord::new(4));
        assert_eq!(term.stdout, vec![4u8]);
    }
}
