#include <Python.h>
#include <stddef.h>

#include "mcts.h"

struct PythonHandle {
  PyObject *object;

  PythonHandle() : object(nullptr) {
  }

  PythonHandle(PyObject *object_) : object(object_) {
  }

  static PythonHandle copy(PyObject *object) {
    Py_INCREF(object);
    return PythonHandle(object);
  }

  PythonHandle(const PythonHandle &) = delete;

  PythonHandle(PythonHandle &&other) : object(other.object) {
    other.object = NULL;
  }

  PythonHandle &operator=(const PythonHandle &) = delete;

  PythonHandle &operator=(PythonHandle &&other) {
    Py_XDECREF(object);

    object = other.object;
    other.object = nullptr;

    return *this;
  }

  void steal() {
    object = NULL;
  }

  ~PythonHandle() {
    Py_XDECREF(object);
  }

  bool null() const {
    return object == NULL;
  }
};

static bool assert_tuple_length(PyObject *tuple, size_t length) {
  if (PyTuple_Check(tuple) && (size_t)PyTuple_Size(tuple) == length) {
    return true;
  }

  PyErr_Format(
      PyExc_TypeError, "expected a tuple of length %u", (unsigned int)length);

  return false;
}

class Trainer {
  const size_t n_games;
  const size_t n_evaluations;
  PyObject *expander;

  std::vector<MCTS<PythonHandle, PythonHandle>> trees;
  std::vector<MCTS<PythonHandle, PythonHandle>::Leaf> leaves;
  std::vector<size_t> leaf_indices;
  std::vector<MCTS<PythonHandle, PythonHandle>::ExpansionEntry> expansion;

  PythonHandle collect_leaves() {
    leaf_indices.clear();

    PythonHandle leaf_states(PyList_New(0));

    if (leaf_states.null()) {
      return NULL;
    }

    for (size_t i = 0; i < n_games; i++) {
      leaves[i] = trees[i].select_leaf();

      if (!leaves[i].present()) {
        continue;
      }

      leaf_indices.push_back(i);

      PyObject *leaf_state = leaves[i].game_state().object;
      Py_INCREF(leaf_state);

      PyList_Append(leaf_states.object, leaf_state);
    }

    return leaf_states;
  }

  bool expand_leaves(PythonHandle leaf_states) {
    PythonHandle expander_args(Py_BuildValue("(N)", leaf_states.object));

    if (expander_args.null()) {
      return false;
    }

    leaf_states.steal();

    PythonHandle expander_return_value(
        PyObject_CallObject(expander, expander_args.object));

    if (expander_return_value.null()) {
      return false;
    }

    PythonHandle iterator(PyObject_GetIter(expander_return_value.object));

    if (iterator.null()) {
      return false;
    }

    for (size_t i = 0;; i++) {
      PythonHandle av_and_expansion_seq(PyIter_Next(iterator.object));

      if (av_and_expansion_seq.null()) {
        if (i < leaf_indices.size()) {
          PyErr_SetString(PyExc_TypeError,
                          "too few values in returned sequence");
          return false;
        }

        break;
      }

      if (i >= leaf_indices.size()) {
        PyErr_SetString(PyExc_TypeError,
                        "too many values in returned sequence");
        return false;
      }

      double av;
      PyObject *expansion_seq;

      if (!assert_tuple_length(av_and_expansion_seq.object, 2) ||
          !PyArg_ParseTuple(
              av_and_expansion_seq.object, "dO", &av, &expansion_seq)) {
        return false;
      }

      if (!collect_expansion(expansion_seq)) {
        return false;
      }

      const size_t index = leaf_indices[i];

      trees[index].expand_leaf(leaves[index], av, std::move(expansion));
    }

    return true;
  }

  bool collect_expansion(PyObject *expansion_seq) {
    expansion.clear();

    PythonHandle iterator(PyObject_GetIter(expansion_seq));

    if (iterator.null()) {
      return false;
    }

    for (;;) {
      PythonHandle expansion_entry(PyIter_Next(iterator.object));

      if (expansion_entry.null()) {
        break;
      }

      PyObject *move;
      PyObject *game_state;
      double prior_probability;

      if (!assert_tuple_length(expansion_entry.object, 3) ||
          !PyArg_ParseTuple(expansion_entry.object,
                            "OOd",
                            &move,
                            &game_state,
                            &prior_probability)) {
        return false;
      }

      MCTS<PythonHandle, PythonHandle>::ExpansionEntry entry = {
          PythonHandle::copy(move),
          PythonHandle::copy(game_state),
          prior_probability};

      expansion.emplace_back(std::move(entry));
    }

    if (PyErr_Occurred()) {
      return false;
    }

    return true;
  }

public:
  Trainer(size_t n_games_,
          double c_init,
          double c_base,
          size_t n_evaluations_,
          PyObject *initial_state_,
          PyObject *expander_)
      : n_games(n_games_), n_evaluations(n_evaluations_), expander(expander_),
        leaves(n_games_) {
    trees.reserve(n_games_);

    for (size_t i = 0; i < n_games_; i++) {
      trees.emplace_back(c_init,
                         c_base,
                         PythonHandle::copy(initial_state_),
                         PythonHandle::copy(Py_None));
    }
  }

  PyObject *train() {
    for (size_t i = 0; i < n_evaluations; i++) {
      PythonHandle leaf_states = collect_leaves();

      if (leaf_states.null()) {
        return NULL;
      }

      if (!expand_leaves(std::move(leaf_states))) {
        return NULL;
      }
    }

    return NULL;
  }
};

static PyObject *play_training_games(PyObject *args, PyObject *kwargs) {
  unsigned long n_games;
  double c_init;
  double c_base;
  unsigned long n_evaluations;
  PyObject *initial_state;
  PyObject *expander;

  static char n_games_str[] = "n_games";
  static char c_init_str[] = "c_init";
  static char c_base_str[] = "c_base";
  static char n_evaluations_str[] = "n_evaluations";
  static char initial_state_str[] = "initial_state";
  static char expander_str[] = "expander";

  static char *keyword_names[] = {n_games_str,
                                  c_init_str,
                                  c_base_str,
                                  n_evaluations_str,
                                  initial_state_str,
                                  expander_str,
                                  NULL};

  if (!PyArg_ParseTupleAndKeywords(args,
                                   kwargs,
                                   "kddkO",
                                   keyword_names,
                                   &n_games,
                                   &c_init,
                                   &c_base,
                                   &n_evaluations,
                                   &initial_state,
                                   &expander)) {
    return NULL;
  }

  try {
    return Trainer(
               n_games, c_init, c_base, n_evaluations, initial_state, expander)
        .train();
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }
}

static PyMethodDef module_methods[] = {{"play_training_games",
                                        play_training_games,
                                        METH_VARARGS | METH_KEYWORDS,
                                        NULL},
                                       {NULL, NULL, 0, NULL}};

static PyModuleDef module_defn = {PyModuleDef_HEAD_INIT,
                                  "a3mcts",
                                  NULL,
                                  0,
                                  module_methods,
                                  NULL,
                                  NULL,
                                  NULL,
                                  NULL};

extern "C" PyObject *PyInit_a3mcts(void) {
  return PyModule_Create(&module_defn);
}
