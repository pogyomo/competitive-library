use std::rc::Rc;

struct PersistentStackNode<T> {
    value: T,
    next: Option<Rc<PersistentStackNode<T>>>,
}

pub struct PersistentStack<T> {
    head: Option<Rc<PersistentStackNode<T>>>,
}

impl<T> PersistentStack<T> {
    pub fn new() -> Self {
        Self { head: None }
    }

    pub fn top(&self) -> Option<&T> {
        Some(&self.head.as_ref()?.value)
    }

    pub fn push(&self, value: T) -> Self {
        Self {
            head: Some(Rc::new(PersistentStackNode {
                value,
                next: self.head.as_ref().map(|node| Rc::clone(node)),
            })),
        }
    }

    pub fn pop(&self) -> Self {
        Self {
            head: self
                .head
                .as_ref()
                .map(|node| node.next.as_ref().map(|node| Rc::clone(node)))
                .flatten(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::PersistentStack;

    #[test]
    fn push_and_pop() {
        let s1 = PersistentStack::new();
        let s2 = s1.push(10);
        let s3 = s2.push(20);
        let s4 = s3.pop();

        assert_eq!(s1.top(), None);
        assert_eq!(s2.top(), Some(&10));
        assert_eq!(s3.top(), Some(&20));
        assert_eq!(s4.top(), Some(&10));
    }
}
