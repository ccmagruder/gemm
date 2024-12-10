#pragma once

#include <memory>


// Use cases
// init on host, send to device
// init on device, send to host

class Matrix {
 private: 
    Matrix(size_t m_, size_t n_) 
		: m(m_)
		, n(n_)
		, _host_ptr(nullptr)
		, _device_ptr(nullptr) 
	{}

 public:
	void toHost(); // allocate if necessary, then copy
	void toDevice(); // allocate if necessary, then copy

	static std::unique_ptr<Matrix> makeHost(size_t m, size_t n); // allocate on host
	static std::unique_ptr<Matrix> makeDevice(size_t m, size_t n); // allocate on device
	// These all construct on the host
	static std::unique_ptr<Matrix> fill(size_t m, size_t n, float value);
	static std::unique_ptr<Matrix> normalIID(size_t m, size_t n);

	const float* const getDevicePtr() const;
	const float* const getHostPtr() const;
	
	float* getDevicePtr();
	float* getHostPtr();
	
	~Matrix(); // free _device_ptr if not null
	// Both access the host memory
	// What happens if host memory not allocated? -> throw
	const float operator() (size_t i, size_t j) const; // get the ij entry
	float& operator() (size_t i, size_t j);

 //// Data
 public: 
	const size_t m;
    const size_t n;

 protected:
    std::unique_ptr<float[], std::default_delete<float[]>> _host_ptr;
	float* _device_ptr;	

 public:

	// TODO: remove me
    size_t m() const { return this->m; }
    size_t n() const { return this->n; }

};

