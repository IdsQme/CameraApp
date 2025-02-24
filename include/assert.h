#include <cstdlib>

#define cassert(expr, message) \
    if (!(expr)) { \
        std::cerr << "Assertion failed: " << (message) << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    }