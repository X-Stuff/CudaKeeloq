#pragma once

#include "common.h"

#include <vector>
#include <cuda_runtime_api.h>

//
template<uint8_t NumDigits> struct MultibaseNumber;

//
using Multibase8Digits = MultibaseNumber<sizeof(uint64_t)>;


/**
 *  This represents a number where each digit has it's one base
 * Imagine this as set of rolling cylinders, set side-by-side
 *
 * Since application of this struct is only byte bruteforce - the maximum allowed base is 255
 *
 * e.g. NumDigits == 4
 *  So there are 4 cylinders
 *  For example 1st cylinder has base 2, second - 4, third - 6, fourth - 8
 *  So the number 1 will be equal 0001
 *  number 10 is equals 0012    (1 * 8) + 2 = | 0 | 0 | 1 | 2 | )
 *  number 100 is       0204   (12 * 8) + 4 = | 0 | 2 | 0 | 4 | )
 *  number 500 is       2224   (62 * 8) + 4 = | 2 | 2 | 2 | 4 | ) = ((10 * 6 + 2) * 8) + 4 = ((((2 * 4 + 2)) * 6 + 2) * 8) + 4
 *
 *  Since this structure is pretty heavy
 * The idea is NOT allow non-const method on device
 * You should have a single const reference on device and
 * call operator+ in order to get new value.
 */
template<uint8_t NumDigits = 8>
struct MultibaseNumber
{
	static_assert(NumDigits <= 8, "At the moment we only support 8 bytes numbers");

	/**
	 *  A generic case when all digits has different numerals and perhaps different bases
	 */
	__host__ MultibaseNumber(const std::vector<uint8_t> numerals[NumDigits]);

	// A special clone constructor
	__host__ MultibaseNumber(const MultibaseNumber& instance, uint64_t number)
	{
		memcpy(Digits, instance.Digits, sizeof(Digits));
		set(number);
	}

	// It's pretty heavy struct if you want clone it - constructor above
	// TODO: disable copy
	// MultibaseNumber(const MultibaseNumber& other) = delete;
	// MultibaseNumber& operator=(const MultibaseNumber& other) = delete;

protected:

	struct ByteDigit;

	/**
	 *  Use the same ByteDigit for every digit in this value
	 */
	__host__ MultibaseNumber(const ByteDigit& digit);

public:

	// returns a 8-bytes representation number for current set of digits
	__host__ __device__ inline uint64_t value() const;

	// returns a 8-bytes representation number for specified @input
	//  This operation will do lookup in ByteDigits for each byte in @input
	//  if byte is not represent in ByteDigit - it will became the first one
	__host__ __device__ inline uint64_t convert(uint64_t input) const;

	//
	__host__ __device__ inline size_t invariants() const;

	template<typename Operand>
	__host__ __device__ inline uint64_t operator+(Operand right) const;

public:
	// updates internal digits of this number.
	// sets them to state as shown in description of this struct
	// e.g:
	//  you want to set number `0x8D 1F 00 78 1E AD 3B 12`
	//   digit[0] has: { 0x10, 0x11, 0x12, 0x13 }
	//   digit[1] has: { 0x3C, 0x3B }
	//   digit[2] has: { 0x10 }
	//   digit[3] has: { 0x10, 0xAA, 0xDE, 0xAC, 0x12 }
	//    etc.
	//
	//   So the result will be:
	//   digit[0].value == 2, digit[0].numeral() == 0x12  -- easy 0x12 was looked up and found
	//   digit[1].value == 1, digit[1].numeral() == 0x3b  -- same 0x3B lookup ok
	//   digit[2].value == 0, digit[2].numeral() == 0x10  -- the only allowed value - input ignored
	//   digit[3].value == 0, digit[3].numeral() == 0x10  -- byte 0x1E wasn't found value set to 0
	//
	// IMPORTANT: LITTLE ENDIAN
	__host__ inline MultibaseNumber& set(uint64_t value);

	// increment internal digits' values by base10value
	// WARNING: overflow possible, will be ignored
	__host__ inline MultibaseNumber& add(uint64_t base10value);

protected:

	// Imagine this as one cylinder
	struct ByteDigit
	{
		friend struct MultibaseNumber<NumDigits>;

		// Creates digit
		__host__ ByteDigit(const std::vector<uint8_t>& numerals);

		// adds @base10num with digit
		// ** returns digit's numeral **
		// sets carry to the @base10num
		template<typename Operand>
		__host__ __device__ static inline uint8_t add_set_carry(const ByteDigit& digit, Operand& base10num);

		// Return numeral by index
		__host__ __device__ inline uint8_t numeral(uint8_t in_index) const;
		// Return numeral of the digit
		//  e.g. your base is 4, your numerals are: 0x23, 0x44, 0xBA, 0xA6
		//  your value is: 0 - numeral() == 0x23, value is: 3 - numeral() == 0xA6
		__host__ __device__ inline uint8_t numeral() const { return numeral(index); }

		// Return numeral by value through lookup
		__host__ __device__ inline uint8_t numeral_lookup(uint8_t value) const { return numeral(lut[value]); }

		// return count of possible numerals for that digit
		__host__ __device__ inline uint8_t count() const { return size; }

	public:

		// RETURNS CARRY
		// It will change internal value and returns carry:
		//  imagine this a how many full cycles
		//  this cylinder made when it was asked to add @base10num
		template<typename Operand>
		__host__  inline Operand add(Operand base10num);

		// this will set the value of digit according to lookup table
		// in case of invalid value (not represented in numerals[] - the value will be 0
		// which means the first numeral)
		__host__ inline void set(uint8_t index);

	private:
		ByteDigit() = default;

		// numeral values. it may be not just 0,1,2,3,4...
		// but for base 4 it may be: 0xA3, 0xCC, 0x01, 0x22
		uint8_t num[0xFF] = {0};

		// lookup table:
		//  at index that equals numeral value there is a value which represents index in numerals
		//  e.g.                              https://asciiflow.com/
		//                                  ┌───────────────────────┐
		//                                  ▲                       │
		// numerals = [ 0x03, 0x02, 0x01, 0x00, ... garbage. ]      │
		//                                  ▲                       │
		//               ┌──────────────────┘                       │
		//               ▲                                          │
		//	    lut = [ 0x03, 0x02, 0x01, 0x00, 0x00 ... 0x00]      │
		//               ▲                                          │
		//               └──────────────────────────────────────────┘
		//
		uint8_t lut[0xFF] = {0};

		// The index in num array which represent exact numeral this digit is
		uint8_t index = 0;

		// Actual size of numerals (the base if number representing by this digit)
		uint8_t size = 0;
	};


protected:

	// the digits
	ByteDigit Digits[NumDigits];
};

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ size_t MultibaseNumber<NumDigits>::invariants() const
{
	size_t num = Digits[0].size;

	UNROLL
	for (uint8_t i = 1; i < NumDigits; ++i)
	{
		num *= Digits[i].size;
	}

	return num;
}

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ uint64_t MultibaseNumber<NumDigits>::value() const
{
	uint64_t result = 0;
	uint8_t* pResult = reinterpret_cast<uint8_t*>(&result);

	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		pResult[i] = Digits[i].numeral();
	}

	return result;
}

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ uint64_t MultibaseNumber<NumDigits>::convert(uint64_t input) const
{
	uint64_t result = 0;
	uint8_t* pResult = reinterpret_cast<uint8_t*>(&result);
	uint8_t* pInput = reinterpret_cast<uint8_t*>(&input);

	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		pResult[i] = Digits[i].numeral_lookup(pInput[i]);
	}
}

template<uint8_t NumDigits /*= 8*/>
template<typename Operand>
__host__ __device__ uint64_t MultibaseNumber<NumDigits>::operator+(Operand right) const
{
	uint64_t result = 0;
	uint8_t* pResult = reinterpret_cast<uint8_t*>(&result);

	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		pResult[i] = ByteDigit::add_set_carry(Digits[i], right);
	}

	return result;
}

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ uint8_t MultibaseNumber<NumDigits>::ByteDigit::numeral(uint8_t in_index) const
{
	assert(in_index < size);

	// Important for optimization purposes
	// WE ARE NOT USING (value % size)
	return num[in_index];
}


template<uint8_t NumDigits /*= 8*/>
template<typename Operand>
__host__ __device__ uint8_t MultibaseNumber<NumDigits>::ByteDigit::add_set_carry(const ByteDigit& digit, Operand& base10num)
{
	// e.g.
	//  digit.num   : { 0x12, 0x45, 0xba, 0xf5, 0xd2 }
	//  digit.index : 2
	// base10num = 142
	//
	// (142 + 2) % 5 = 4
	// (142 + 2) / 5 = 28
	//
	// result:
	//  digit.index = 4
	//  base10num = 28
	uint8_t new_digit_index = (uint8_t)((base10num + digit.index) % digit.size);

	base10num = (base10num + digit.index) / digit.size;

	return digit.num[new_digit_index];
}


template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseNumber<NumDigits>::MultibaseNumber(const std::vector<uint8_t> numerals[NumDigits])
{
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		Digits[i] = ByteDigit(numerals[i]);
	}
}

template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseNumber<NumDigits>::MultibaseNumber(const ByteDigit& digit)
{
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		Digits[i] = digit;
	}
}



template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseNumber<NumDigits>::ByteDigit::ByteDigit(const std::vector<uint8_t>& numerals)
{
	// default
	index = 0;

	// incrementing in the loop, for the duplicate numerals cases
	size = 0;

	for (uint8_t i = 0; i < numerals.size(); ++i)
	{
		// Getting next numeral candidate
		uint8_t numeral_value = numerals[i];

		// if there is 0 value (index) in the lookup table
		// that means `numeral_value` wasn't added to available values yet
		if (!lut[numeral_value])
		{
			// putting index of `numeric_value` to the lut
			lut[numeral_value] = size;

			// setting the size-th numeral
			num[size] = numeral_value;

			// increasing the size
			++size;
		}
	}

	assert(size > 0 && "Digit base should be at least 0");
}

template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseNumber<NumDigits>& MultibaseNumber<NumDigits>::set(uint64_t index)
{
	uint8_t* pValue = reinterpret_cast<uint8_t*>(&index);

	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		Digits[i].set(pValue[i]);
	}

	return *this;
}

template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseNumber<NumDigits>& MultibaseNumber<NumDigits>::add(uint64_t base10value)
{
	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		base10value = Digits[i].add(base10value);
	}

	// here base10value will contain overflow
	// not sure what to do with it

	return *this;
}


template<uint8_t NumDigits /*= 8*/>
template<typename Operand>
__host__ Operand MultibaseNumber<NumDigits>::ByteDigit::add(Operand base10num)
{
	index = (uint8_t)( (index + base10num) % size);
	return (index + base10num) / size;
}

template<uint8_t NumDigits /*= 8*/>
__host__ void MultibaseNumber<NumDigits>::ByteDigit::set(uint8_t v)
{
	index = lut[v];
}
