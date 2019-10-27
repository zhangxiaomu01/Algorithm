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



//1115. Print FooBar Alternately
//https://leetcode.com/problems/print-foobar-alternately/
class FooBar {
private:
    int n;
    mutex mu;
    condition_variable cv;
    bool isFoo;

public:
    FooBar(int n) {
        this->n = n;
        isFoo = false;
    }

    void foo(function<void()> printFoo) {
        //we cannot put unique_lock here, or when the function call ends
        //we will potentially unlock locker twice!
        //unique_lock<mutex> locker(mu);
        
        for (int i = 0; i < n; i++) {
            //Define locker here, note when we notify other's we set the 
            //state to be another one
            unique_lock<mutex> locker(mu);
            while(isFoo){
                cv.wait(locker);
            }
            // printFoo() outputs "foo". Do not change or remove this line.
            printFoo();
            isFoo = true;
            cv.notify_one();
            
        }
    }

    void bar(function<void()> printBar) {
        for (int i = 0; i < n; i++) {
            
            unique_lock<mutex> locker(mu);
            while(!isFoo){
                cv.wait(locker);
            }
            // printBar() outputs "bar". Do not change or remove this line.
            printBar();
            isFoo = false;
            
            cv.notify_one();
            
        }
    }
};


//1226. The Dining Philosophers
//https://leetcode.com/problems/the-dining-philosophers/
//Not easy to get the solution. Hard to understand the problem!
class DiningPhilosophers {
private:
    mutex mu_[5];
public:
    DiningPhilosophers() {
        
    }

    void wantsToEat(int philosopher,
                    function<void()> pickLeftFork,
                    function<void()> pickRightFork,
                    function<void()> eat,
                    function<void()> putLeftFork,
                    function<void()> putRightFork) {
		int l = philosopher;
        int r = (philosopher + 1) % 5;
        //the person with even index always grab right fork first
        if(philosopher % 2 == 0){
            mu_[r].lock();
            mu_[l].lock();
            pickRightFork();
            pickLeftFork();
            eat();
            putLeftFork();
            putRightFork();
            mu_[l].unlock();
            mu_[r].unlock();
            
        }else{
            mu_[l].lock();
            mu_[r].lock();
            pickLeftFork();
            pickRightFork();
            eat();
            putRightFork();
            putLeftFork();
            mu_[r].unlock();
            mu_[l].unlock();
        } 
        
    }
};
