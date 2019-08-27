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


//////////////////////////////////////////////////////////////////////////////
//Design cash register program
#include<iostream>
#include<vector>
#include<queue>
#include<map>
#include<iterator>
#include<sstream>

/*
Recursive solution + memorization
*/

class CashRegister
{
public:
	CashRegister() = default;
	CashRegister(const std::vector<int>& num) {
		
		for (int i = 0;i < num.size(); ++i) {
			mTill[denomination(enumMaps[i])] = num[i];
		}
		//Initialize total money
		m_totalMoney = static_cast<int> (TotalInRegister() * 100);
	};
	~CashRegister() {};

	// the customer has paid money,
	// that money is already in the till
	// Now, dispense change
	//Assume the amountPaid and amountOwed represents dollars (not cents)
	void MakeChange(double amountPaid, double amountOwed);

private:
	// the cash register holds zero or more of these
	// bills and coins in the Till.
	// The value of the enum is its worth in cents
	enum denomination
	{
		kPenny = 1,
		kNickel = 5,
		kDime = 10,
		kQuarter = 25,
		kHalf = 50,
		kDollar = 100,
		kFive = 500,
		kTen = 1000,
		kTwenty = 2000,
		kFifty = 5000,
		kHundred = 10000
	};


	// how much money is in the cash register?
	double TotalInRegister() const;

	// remove coins/bills from the till and give them to the customer 
	//--------------------Did not use it in my code --------------------------
	void Dispense(denomination d, int count);

	// there is a problem!
	void ReportError(const char *text) const;

	//Recursive helper function
	int helper(std::vector<std::vector<int>>& memo, int amount, int pos);

private:

	// This is the till.  All bills and coins are stored here
	std::map< denomination, int > mTill;

	// This is the LCD display on the cash register
	std::ostringstream mDisplay;

	//The totalNumber of money (cents). 
	//The range may not be sufficient for large transaction!
	int m_totalMoney;
	
	//A map between enum and array
	int enumMaps[11] = { 1, 5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000 };

};

// -------------------------------------------------------
// Function:    CashRegister::TotalInRegister
// Purpose:     how much money is in the cash register?
double CashRegister::TotalInRegister() const
{
	int total(0);
	auto it = mTill.begin();
	for (; it != mTill.end(); it++)
		total += ((int)it->first) * it->second;
	std::cout << "Total maoney in register: " << total / 100.0 << " dollars" << std::endl;
	return total / 100.0;
}

// -------------------------------------------------------
// Function:    CashRegister::Dispense
// Purpose:     remove coins/bills from the till and give them to the customer

void CashRegister::Dispense(denomination d, int count)
{
	mTill[d] -= count;
}

// -------------------------------------------------------
// Function:    CashRegister::ReportError
// Purpose:     there is a problem!

void CashRegister::ReportError(const char *text) const
{
	// show the problem on the display 
	//mDisplay << text;
	std::cout << text << std::endl;
}

// -------------------------------------------------------

// ******************************
//     <insert your code here>
// ******************************

int CashRegister::helper(std::vector<std::vector<int>>& memo, int amount, int pos) {
	//int enumMaps[11] = { 1, 5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000 };
	if (amount == 0) return 1;
	if (amount < 0 || pos < 0) return 0;
	if (memo[amount][pos] != -1) return memo[amount][pos];
	//always start from the largest possible money
	for (int i = pos; i >= 0; --i) {
		denomination d = denomination(enumMaps[i]);
		int numOfCoins = mTill[denomination(enumMaps[i])];
		int coinDenomination = enumMaps[i];
		if (numOfCoins > 0 && coinDenomination <= amount) {
			mTill[d]--;
			if (helper(memo, amount - coinDenomination, i) == 1) {
				memo[amount][pos] = 1;
				return memo[amount][pos];
			}
			mTill[d]++;
		}
	}
	memo[amount][pos] = 0;
	return memo[amount][pos];
}

void CashRegister::MakeChange(double amountPaid, double amountOwed)
{
	int difference = static_cast<int> (amountPaid * 100.0 - amountOwed * 100.0);

	const char* msg = "No sufficient funds in the register. ";
	if(difference > m_totalMoney){
		ReportError(msg);
		return;
	}

	std::vector<std::vector<int>> memo(difference+1, std::vector<int>(11, -1));
	//Make a copy of the mTill, so we can track which coin we need to pick up later
	std::map< denomination, int > coinsLeft(mTill.begin(), mTill.end());

	if (helper(memo, difference, 10)) {
		std::vector<int> coinsToPick(11, 0);
		int i = 0;
		auto coinsLeftIt = coinsLeft.begin();
		for (auto it = mTill.begin(); it != mTill.end(); ++it) {
			coinsToPick[i++] = coinsLeftIt->second - it->second;
			coinsLeftIt++;
		}
		//May be better to wrap in the function, print one possible choice
		std::cout << "One of the possible choice for the change is:" << std::endl;
		std::cout << "kPenny:   " << coinsToPick[0] << std::endl;
		std::cout << "kNickel:  " << coinsToPick[1] << std::endl;
		std::cout << "kDime:    " << coinsToPick[2] << std::endl;
		std::cout << "kQuarter: " << coinsToPick[3] << std::endl;
		std::cout << "kHalf:    " << coinsToPick[4] << std::endl;
		std::cout << "kDollar:  " << coinsToPick[5] << std::endl;
		std::cout << "kFive:    " << coinsToPick[6] << std::endl;
		std::cout << "kTen:     " << coinsToPick[7] << std::endl;
		std::cout << "kTwenty:  " << coinsToPick[8] << std::endl;
		std::cout << "kFifty:   " << coinsToPick[9] << std::endl;
		std::cout << "kHundred: " << coinsToPick[10] << std::endl;
	}
	else {
		ReportError("Impossible to make the change! \n");
	}
	m_totalMoney = static_cast<int>(TotalInRegister());
	
}


int main() {
	int enumMaps[11] = { 1, 5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000 };
	std::vector<int> numOfCoins = { 3,11,23,4,12,6,9,4,7,12,1 };
	CashRegister cR(numOfCoins);
	//Assume the amountPaid and amountOwed represents dollars (not cents)
	//We will modify the mTill when we call the function
	cR.MakeChange(19.73, 14.28); //Good
	//cR.MakeChange(50.05, 0.01);//impossible to make the change
	//cR.MakeChange(1000.0, 14.28);//No sufficient funds

	system("pause");
	return 0;
}


