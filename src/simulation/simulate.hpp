#if !defined(KANACHAN_SIMULATION_SIMULATE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_SIMULATE_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <string>
#include <memory>


namespace Kanachan{

boost::python::list simulate(
  boost::python::object device,
  boost::python::object dtype,
  long room,
  long baseline_grade,
  boost::python::object baseline_model,
  long proposed_grade,
  boost::python::object proposed_model,
  long simulation_mode,
  long num_simulation_sets,
  long batch_size,
  long concurrency,
  boost::python::object progress,
  boost::python::object seed = boost::python::object());

std::shared_ptr<Kanachan::GameLog> test(
  boost::python::long_ simulation_mode,
  boost::python::tuple grades,
  boost::python::object test_model,
  boost::python::list test_paishan_list);

} // namespace Kanachan


BOOST_PYTHON_MODULE(_simulation)
{
  boost::python::class_<
    Kanachan::GameLog, std::shared_ptr<Kanachan::GameLog>, boost::noncopyable
  >("GameLog", boost::python::no_init)
    .def("get_result", &Kanachan::GameLog::getResult)
    .def("get_episode", &Kanachan::GameLog::getEpisode)
    .def("get_metadata", &Kanachan::GameLog::getMetadata);
  namespace python = boost::python;
  python::def("simulate", &Kanachan::simulate,
    (python::arg("device"), python::arg("dtype"), python::arg("room"),
     python::arg("baseline_grade"), python::arg("baseline_model"),
     python::arg("proposed_grade"), python::arg("proposed_model"),
     python::arg("simulation_mode"), python::arg("num_simulation_sets"),
     python::arg("batch_size"), python::arg("concurrency"),
     python::arg("progress"), python::arg("seed") = python::object()));
  boost::python::def("test", &Kanachan::test);
} // BOOST_PYTHON_MODULE(_simulation)


#endif // !defined(KANACHAN_SIMULATION_SIMULATE_HPP_INCLUDE_GUARD)
