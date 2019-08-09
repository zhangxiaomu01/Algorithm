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


/////////////////////////////////////////////////////////////////////////
//Read code problem
#include <exception>

class container_base
{
protected:
	float * pData;
	size_t size;
	size_t count;
	
	void grow()
	{
		if (size == 0)
			throw std::exception();

		size *= 2;
		float * newBuffer = new float[size];

		for (size_t i = 0; i < count; ++i)
			newBuffer[i] = pData[i];

		delete[] pData;
		pData = newBuffer;
	}

public:
	container_base()
	{
		size = 100;
		pData = new float[size]; // Start with an arbitrary size
		count = 0;
	}

	~container_base()
	{
		delete[] pData;
	}

	bool empty() const
	{
		return count == 0;
	}

	void clear()
	{
		count = 0;
	}

	container_base(const container_base & source)
	{
		if (source.size > this->size)
		{
			delete[] this->pData;
			this->pData = new float[source.size];
		}
		for (size_t i = 0; i < source.size; ++i)
			this->pData[i] = source.pData[i];
		this->size = source.size;
		this->count = source.count;
	}

	container_base(container_base && source)
	{
		delete[] this->pData;
		this->pData = source.pData;
		this->size = source.size;
		this->count = source.count;

		source.pData = nullptr;
		source.size = 0;
		source.count = 0;
	}

	void add(float value)
	{
		if (count >= size)
			grow();

		pData[count++] = value;
	}

};


class container1 : public container_base
{
public:
	float get()
	{
		if (empty())
			throw std::exception();
		--count;
		return pData[count];
	}
};

class container2 : public container_base
{
public:
	float get(size_t index) const
	{
		if (empty() || index >= count)
			throw std::exception();
		return pData[index];
	}
};

class container3
{
	float * pData;
	size_t size;
	size_t index_first;
	size_t index_last;

	static size_t copy_to_buffer(float * pDest, const container3 & source)
	{
		const size_t len = source.length();
		size_t end;
		if (source.index_first < source.index_last)
			end = source.index_last;
		else
			end = source.size;
		size_t rx_index, tx_index = 0;
		for (rx_index = source.index_first; rx_index < end; ++rx_index)
			pDest[tx_index++] = source.pData[rx_index];
		if (source.index_last < source.index_first)
		{
			// There is wrap-around			
			for (rx_index = 0; rx_index < source.index_last; ++rx_index)
				pDest[tx_index++] = source.pData[rx_index];
		}
		return len;
	}

	void grow()
	{
		if (size == 0)
			throw std::exception();

		size_t newSize = size * 2;
		float * newBuffer = new float[newSize];

		size_t len = copy_to_buffer(newBuffer, *this);

		delete[] pData;
		pData = newBuffer;
		size = newSize;
		index_first = 0;
		index_last = len;
	}

	// Need to leave one space vacant, otherwise the full and empty states are the same. 
	inline size_t capacity() const { return size - 1; }

public:
	container3()
	{
		size = 100;
		pData = new float[size]; // Start with an arbitrary size
		index_first = 0;
		index_last = 0;
	}

	~container3()
	{
		delete[] pData;
	}

	bool empty() const
	{
		return length() == 0;
	}

	void clear()
	{
		index_first = 0;
		index_last = 0;
	}

	size_t length() const
	{
		if (index_last >= index_first)
			return index_last - index_first;
		else
		{
			size_t len = (size - index_first) + index_last;
			if (len >= size)
				throw std::exception();
			return len;
		}
	}

	container3(container3 const & source)
	{
		if (source.size > this->size)
		{
			delete[] this->pData;
			this->pData = new float[source.size];
		}
		size_t len = copy_to_buffer(this->pData, source);
		this->index_first = 0;
		this->index_last = len;
	}

	container3(container3 && source)
	{
		delete[] this->pData;
		this->pData = source.pData;
		this->size = source.size;
		this->index_first = source.index_first;
		this->index_last = source.index_last;

		source.pData = nullptr;
		source.size = 0;
		source.index_first = 0;
		source.index_last = 0;
	}

	void add(float value)
	{
		if (length() >= capacity())
			grow();
		if (index_last == size)
			index_last = 0; // wrap around
		pData[index_last++] = value;
	}

	float get()
	{
		if (empty())
			throw std::exception();
		size_t current_index = index_first++;
		index_first %= size;					// Handle wrap-around case.
		return pData[current_index];
	}
};




