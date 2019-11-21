#include <Python.h>
#include <stddef.h>

#include <cstdio> // fixme

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

  PyObject *steal() {
    PyObject *object_ = object;
    object = NULL;
    return object_;
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

  PyErr_Format(PyExc_TypeError, "expected a tuple of length %u", (unsigned int)length);

  return false;
}

class Trainer {
  const size_t n_games;
  const size_t n_evaluations;
  PyObject *expand;

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

      if (PyList_Append(leaf_states.object, leaf_state) < 0) {
        Py_DECREF(leaf_state);
        return NULL;
      }
    }

    return leaf_states;
  }

  bool expand_leaves(PythonHandle leaf_states) {
    PythonHandle expand_args(Py_BuildValue("(N)", leaf_states.object));

    if (expand_args.null()) {
      return false;
    }

    leaf_states.steal();

    PythonHandle expand_return_value(PyObject_CallObject(expand, expand_args.object));

    if (expand_return_value.null()) {
      return false;
    }

    PythonHandle iterator(PyObject_GetIter(expand_return_value.object));

    if (iterator.null()) {
      return false;
    }

    for (size_t i = 0;; i++) {
      PythonHandle av_and_expansion_seq(PyIter_Next(iterator.object));

      if (av_and_expansion_seq.null()) {
        if (PyErr_Occurred()) {
          return false;
        }

        if (i < leaf_indices.size()) {
          PyErr_SetString(PyExc_TypeError, "too few values in returned sequence");
          return false;
        }

        break;
      }

      if (i >= leaf_indices.size()) {
        PyErr_SetString(PyExc_TypeError, "too many values in returned sequence");
        return false;
      }

      double av;
      PyObject *expansion_seq;

      if (!assert_tuple_length(av_and_expansion_seq.object, 2) ||
          !PyArg_ParseTuple(av_and_expansion_seq.object, "dO", &av, &expansion_seq)) {
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
          !PyArg_ParseTuple(
              expansion_entry.object, "OOd", &move, &game_state, &prior_probability)) {
        return false;
      }

      MCTS<PythonHandle, PythonHandle>::ExpansionEntry entry = {
          PythonHandle::copy(move), PythonHandle::copy(game_state), prior_probability};

      expansion.emplace_back(std::move(entry));
    }

    if (PyErr_Occurred()) {
      return false;
    }

    return true;
  }

  PyObject *collect_results() {
    PythonHandle results(PyList_New(n_games));

    if (results.null()) {
      return NULL;
    }

    for (size_t i = 0; i < n_games; i++) {
      auto &&mcts = trees[i];

      while (mcts.expanded() && !mcts.complete()) {
        mcts.move_proportional();
      }

      MCTS<PythonHandle, PythonHandle>::GameResult game_result = std::move(mcts).finalize_result();

      PythonHandle history(PyList_New(game_result.history.size()));

      for (size_t j = 0; j < game_result.history.size(); j++) {
        auto &&game_state = game_result.history[j].game_state;
        auto &&search_probabilities = game_result.history[j].search_probabilities;

        PythonHandle probabilities = PyList_New(search_probabilities.size());

        for (size_t k = 0; k < search_probabilities.size(); k++) {
          auto &&move = search_probabilities[k].first;
          auto &&probability = search_probabilities[k].second;

          PyObject *tuple = Py_BuildValue("Nd", move.object, probability);

          if (tuple == NULL) {
            return NULL;
          }

          move.steal();

          PyList_SET_ITEM(probabilities.object, k, tuple);
        }

        PyObject *history_entry = Py_BuildValue("NN", game_state.object, probabilities.object);

        if (history_entry == NULL) {
          return NULL;
        }

        game_state.steal();
        probabilities.steal();

        PyList_SET_ITEM(history.object, j, history_entry);
      }

      PyObject *result_tuple = Py_BuildValue("dN", game_result.score, history.object);

      if (result_tuple == NULL) {
        return NULL;
      }

      history.steal();

      PyList_SET_ITEM(results.object, i, result_tuple);
    }

    return results.steal();
  }

public:
  Trainer(size_t n_games_,
          size_t n_evaluations_,
          double c_init,
          double c_base,
          PyObject *initial_state_,
          PyObject *expand_)
      : n_games(n_games_), n_evaluations(n_evaluations_), expand(expand_), leaves(n_games_) {
    trees.reserve(n_games_);

    for (size_t i = 0; i < n_games_; i++) {
      size_t seed = 13 + 37 * i * i * i;
      trees.emplace_back(
          c_init, c_base, seed, PythonHandle::copy(initial_state_), PythonHandle::copy(Py_None));
    }
  }

  PyObject *train() && {
    for (size_t i = 0; i < n_evaluations; i++) {
      PythonHandle leaf_states = collect_leaves();

      if (leaf_states.null()) {
        return NULL;
      }

      if (!expand_leaves(std::move(leaf_states))) {
        return NULL;
      }
    }

    return collect_results();
  }
};

static PyObject *play_training_games(PyObject *module, PyObject *args, PyObject *kwargs) {
  (void)module;

  unsigned long n_games;
  double c_init;
  double c_base;
  unsigned long n_evaluations;
  PyObject *initial_state;
  PyObject *expand;

  static char n_games_str[] = "n_games";
  static char n_evaluations_str[] = "n_evaluations";
  static char c_init_str[] = "c_init";
  static char c_base_str[] = "c_base";
  static char initial_state_str[] = "initial_state";
  static char expand_str[] = "expand";

  static char *keyword_names[] = {
      n_games_str, n_evaluations_str, c_init_str, c_base_str, initial_state_str, expand_str, NULL};

  if (!PyArg_ParseTupleAndKeywords(args,
                                   kwargs,
                                   "kkddOO",
                                   keyword_names,
                                   &n_games,
                                   &n_evaluations,
                                   &c_init,
                                   &c_base,
                                   &initial_state,
                                   &expand)) {
    return NULL;
  }

  try {
    Trainer trainer(n_games, n_evaluations, c_init, c_base, initial_state, expand);
    return std::move(trainer).train();
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }
}

static PyMethodDef module_methods[] = {
    {"play_training_games", (PyCFunction)play_training_games, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

static PyModuleDef module_defn = {
    PyModuleDef_HEAD_INIT, "a3mcts", NULL, 0, module_methods, NULL, NULL, NULL, NULL};

extern "C" PyObject *PyInit_a3mcts(void) {
  return PyModule_Create(&module_defn);
}
