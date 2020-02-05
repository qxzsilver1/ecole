#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include "ecole/configuring.hpp"
#include "ecole/observation/node-bipartite.hpp"
#include "ecole/scip/model.hpp"

#include "wrapper/environment.hpp"

namespace py11 = pybind11;

using namespace ecole;

namespace ecole {
namespace configuring {

template <>
void Configure<py11::object>::set(scip::Model& model, action_t const& action) {
	if (py11::isinstance<py11::bool_>(action))
		model.set_param(param, action.cast<scip::param_t<scip::ParamType::Bool>>());
	else if (py11::isinstance<py11::int_>(action))
		// Casting to more precise, and may be downcasted in set_param call
		model.set_param(param, action.cast<scip::param_t<scip::ParamType::LongInt>>());
	else if (py11::isinstance<py11::float_>(action))
		model.set_param(param, action.cast<scip::param_t<scip::ParamType::Real>>());
	else if (py11::isinstance<py11::str>(action)) {
		// Cast as std::string and let set_param do conversion for char
		model.set_param(param, action.cast<std::string>().c_str());
	} else
		// Get exception from set_param
		model.set_param(param, action);
}

}  // namespace configuring
}  // namespace ecole

PYBIND11_MODULE(configuring, m) {
	m.doc() = "Learning to configure task.";
	// Import of abstract required for resolving inheritance to abstract types
	py11::module abstract_mod = py11::module::import("ecole.abstract");

	using ActionFunction = pyenvironment::ActionFunctionBase<configuring::ActionFunction>;
	using Configure = pyenvironment::
		ActionFunction<configuring::Configure<py11::object>, configuring::ActionFunction>;
	using Env = pyenvironment::Env<configuring::Environment>;

	py11::class_<ActionFunction, std::shared_ptr<ActionFunction>>(m, "ActionFunction");
	py11::class_<Configure, ActionFunction, std::shared_ptr<Configure>>(m, "Configure")  //
		.def(py11::init<std::string const&>())
		.def("set", [](Configure& c, scip::Model model, py11::object param) {
			c.set(model, pyenvironment::Action<py11::object>(param));
		});

	py11::class_<Env, pyenvironment::EnvBase>(m, "Environment")  //
		.def_static(
			"make_dummy",
			[](std::string const& param) {
				return std::make_unique<Env>(
					std::make_unique<pyobservation::ObsFunction<observation::NodeBipartite>>(),
					std::make_unique<Configure>(param));
			})
		.def(py11::init(  //
			[](
				pyobservation::ObsFunctionBase const& obs_func,
				ActionFunction const& action_func) {
				return std::make_unique<Env>(obs_func.clone(), action_func.clone());
			}))
		.def("step", [](pyenvironment::EnvBase& env, py11::object const& action) {
			return env.step(pyenvironment::Action<py11::object>(action));
		});
}
