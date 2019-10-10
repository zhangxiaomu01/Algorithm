//1114. Print in Order
//https://leetcode.com/problems/print-in-order/
//Mutex Condition variable!
class Foo {
private:
    mutex mu;
    condition_variable con;
    int cnt;
    
public:
    Foo() {
        cnt = 1;
    }

    void first(function<void()> printFirst) {
        unique_lock<mutex> locker(mu);
        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();
        cnt = 2;
        con.notify_all();
        
    }

    void second(function<void()> printSecond) {
        unique_lock<mutex> locker(mu);
        //keep waiting, prevent spurious wake up
        while(cnt != 2){
            con.wait(locker);
        }
        
        // printSecond() outputs "second". Do not change or remove this line.
        printSecond();
        cnt = 3;
        con.notify_all();
    }

    void third(function<void()> printThird) {
        unique_lock<mutex> locker(mu);
        
        while(cnt != 3){
            con.wait(locker);
        }
        // printThird() outputs "third". Do not change or remove this line.
        printThird();
    }
};


