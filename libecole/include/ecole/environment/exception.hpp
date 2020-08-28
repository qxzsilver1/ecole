#pragma once

#include <exception>
#include <string>

namespace ecole {
namespace environment {

class Exception : public std::exception {
public:
	Exception(std::string message);

	char const* what() const noexcept override;

private:
	std::string message;
};

}  // namespace environment
}  // namespace ecole
