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



//1117. Building H2O
//https://leetcode.com/problems/building-h2o/
class H2O {
private:
    mutex mu;
    condition_variable cv;
    int cnt;
    
public:
    H2O() {
        cnt = 0;
    }

    void hydrogen(function<void()> releaseHydrogen) {
        unique_lock<mutex> locker(mu);
        while(cnt % 3 >= 2){
            cv.wait(locker);
        }
        // releaseHydrogen() outputs "H". Do not change or remove this line.
        releaseHydrogen();
        
        cnt++;
        cv.notify_one();
    }

    void oxygen(function<void()> releaseOxygen) {
        unique_lock<mutex> locker(mu);
        while(cnt % 3 < 2){
            cv.wait(locker);
        }
        // releaseOxygen() outputs "O". Do not change or remove this line.
        releaseOxygen();
        
        cnt++;
        cv.notify_one();
    }
};



