#include <iostream>
void func(int*  a)
{
  ++a;
}

// What would be the output of the following program ?
int main()
{
  int* a = new int;
  *a = 7;
  int& b = *a;
  *a = 10;
  std::cout << "b=" << b << std::endl;

  int d = 24;
  func(&d);
  std::cout << "d=" << d << std::endl;
}

////////////////////////////////////////////////////////////////
#include <iostream>

struct Shape
{
  Shape(){ std::cout << "Shape+" << std::endl; }
  ~Shape(){ std::cout << "Shape-" << std::endl; }
};

struct Circle : public Shape
{
  Circle(){ std::cout << "Circle+" << std::endl; }
  ~Circle(){ std::cout << "Circle-" << std::endl; }
};


// What would be the output of the following program ?
int main() {
  Shape* c = new Circle();
  delete c;
}

///////////////////////////////////////////////////////////////
#include <iostream>
struct A
{
  int a;
  char b;
};

class  B
{
public:
	B() : val(0.0){}
	virtual ~B(){ val = 0.0;}	
private:
	float val;
};

// What would be the output of the following program ?
int main() {
	std::cout << "Size of A:" << sizeof(A) << std::endl;
	std::cout << "Size of B:" << sizeof(B) << std::endl;
}

////////////////////////////////////////////////////////////////////
#include <iostream>

int* getArray()
{
  int data[4] =  { 10, 20, 30, 40 };
  return data;
}
// please review the following code:
int main()
{
	std::cout<< getArray()[0] << std::endl;
}


///////////////////////////////////////////////////////////////////////
#include <cstring>
#include <iostream>

// At a minimum, what changes would you make to the following class:
class String
{
public:
  String(const char* str, int n) : m_data(new char[n]), m_size(n) 
  { std::memcpy(m_data, str, n); }

private:
  char*     m_data;
  int       m_size;
};

int main() {
  {
  String example("aa", 2);
  }
  return 0;
}

//////////////////////////////////////////////////////////
   class Base
    {
    public:
        Base()              { cout << "Base::Base()\n"; }
        virtual ~Base()         { cout << "Base::~Base()\n"; }
        void f()            { cout << "Base::f()\n"; }
        virtual void v()        { cout << "Base::v()\n"; }
    };

    class Derived : public Base
    {
    public:
        Derived()           { cout << "Derived::Derived()\n"; }
        virtual ~Derived()      { cout << "Derived::~Derived()\n"; }
        void f()            { cout << "Derived::f()\n"; }
        virtual void v()        { cout << "Derived::v()\n"; }
    };

    void g(Base base)
    {
        base.v();
    }

    int main()
    {
        Base * d = new Derived();
        /*
        Base::Base()
        Derived::Derived()
        */
        d->f();
        /*
        Base::f()
        */
        d->v();
        /*
        Derived::v()
        */
        g(*d);
        /*
        Base base = *d;
        Base::v()
        Base::~Base()
        */
        delete d;
        /*
        Base::Base()
        Derived::~Derived()
        Base::~Base()
        
        */
        return 0;
    }



