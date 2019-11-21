#include <Python.h>
#include <stddef.h>

#include "mcts.h"

struct PythonHandle {
  PyObject *object;

  PythonHandle() : object(NULL) {
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
    other.object = NULL;

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

struct TypeSpec {
  const char *name;
  size_t size;
  newfunc create;
  destructor destroy;
  PyMethodDef *methods;
};

static PythonHandle create_type(const TypeSpec *spec);

struct PyMCTS {
  PyObject_HEAD MCTS<PythonHandle, PythonHandle> mcts;
};

static PyObject *mcts_create(PyTypeObject *type, PyObject *args, PyObject *kwargs);

static void mcts_destroy(PyObject *self);

static PyObject *mcts_game_state(PyObject *self, PyObject *args);
static PyObject *mcts_expanded(PyObject *self, PyObject *args);
static PyObject *mcts_complete(PyObject *self, PyObject *args);
static PyObject *mcts_collected(PyObject *self, PyObject *args);
static PyObject *mcts_turns(PyObject *self, PyObject *args);
static PyObject *mcts_add_dirichlet_noise(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *mcts_select_leaf(PyObject *self, PyObject *args);
static PyObject *mcts_expand_leaf(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *mcts_move_greedy(PyObject *self, PyObject *args);
static PyObject *mcts_move_proportional(PyObject *self, PyObject *args);
static PyObject *mcts_collect_result(PyObject *self, PyObject *args);
static PyObject *mcts_reset(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef mcts_methods[] = {
    {"game_state", mcts_game_state, METH_NOARGS, NULL},
    {"expanded", mcts_expanded, METH_NOARGS, NULL},
    {"complete", mcts_complete, METH_NOARGS, NULL},
    {"collected", mcts_collected, METH_NOARGS, NULL},
    {"turns", mcts_turns, METH_NOARGS, NULL},
    {"add_dirichlet_noise",
     (PyCFunction)mcts_add_dirichlet_noise,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"select_leaf", mcts_select_leaf, METH_NOARGS, NULL},
    {"expand_leaf", (PyCFunction)mcts_expand_leaf, METH_VARARGS | METH_KEYWORDS, NULL},
    {"move_greedy", mcts_move_greedy, METH_NOARGS, NULL},
    {"move_proportional", mcts_move_proportional, METH_NOARGS, NULL},
    {"collect_result", mcts_collect_result, METH_NOARGS, NULL},
    {"reset", (PyCFunction)mcts_reset, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, -1, NULL}};

const static TypeSpec mcts_typespec = {
    "MCTS", sizeof(PyMCTS), mcts_create, mcts_destroy, mcts_methods};

static PyModuleDef module_defn = {
    PyModuleDef_HEAD_INIT, "a3mcts", NULL, 0, NULL, NULL, NULL, NULL, NULL};

static bool assert_tuple_length(PyObject *tuple, size_t length);

template <class Iterator, class Fn>
static PythonHandle iterator_to_list(Iterator begin, Iterator end, Fn fn) {
  PythonHandle list(PyList_New((Py_ssize_t)(end - begin)));

  auto it = begin;

  for (size_t i = 0; i < (size_t)(end - begin); i++) {
    PythonHandle element(fn(*(it++)));

    if (element.null()) {
      return PythonHandle(NULL);
    }

    PyList_SET_ITEM(list.object, (Py_ssize_t)i, element.steal());
  }

  return list;
}

extern "C" PyObject *PyInit_a3mcts(void) {
  PythonHandle module(PyModule_Create(&module_defn));

  if (module.null()) {
    return NULL;
  }

  PythonHandle mcts_type(create_type(&mcts_typespec));

  if (mcts_type.null()) {
    return NULL;
  }

  if (PyModule_AddObject(module.object, mcts_typespec.name, mcts_type.object) < 0) {
    return NULL;
  }

  mcts_type.steal();

  return module.steal();
}

static PythonHandle create_type(const TypeSpec *spec) {
  void *buffer = PyObject_Malloc(sizeof(PyTypeObject));

  if (buffer == NULL) {
    return PythonHandle(NULL);
  }

  memset(buffer, 0, sizeof(PyTypeObject));
  PyObject_Init((PyObject *)buffer, &PyType_Type);

  PythonHandle type((PyObject *)buffer);

  auto tp = (PyTypeObject *)type.object;
  tp->tp_name = spec->name;
  tp->tp_basicsize = spec->size;
  tp->tp_new = spec->create;
  tp->tp_dealloc = spec->destroy;
  tp->tp_methods = spec->methods;
  tp->tp_flags = Py_TPFLAGS_DEFAULT;

  if (PyType_Ready(tp) < 0) {
    return PythonHandle(NULL);
  }

  return type;
}

static PyObject *mcts_create(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  PyObject *initial_state;
  double c_init;
  double c_base;

  static char c_init_str[] = "c_init";
  static char c_base_str[] = "c_base";
  static char initial_state_str[] = "initial_state";

  static char *keyword_names[] = {c_init_str, c_base_str, initial_state_str, NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "ddO", keyword_names, &c_init, &c_base, &initial_state)) {
    return NULL;
  }

  PythonHandle self(PyObject_New(PyObject, type));

  if (self.null()) {
    return NULL;
  }

  void *location = &((PyMCTS *)self.object)->mcts;

  try {
    new (location) MCTS<PythonHandle, PythonHandle>(
        c_init, c_base, PythonHandle::copy(initial_state), PythonHandle::copy(Py_None));
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  return self.steal();
}

static void mcts_destroy(PyObject *self) {
  auto &mcts = ((PyMCTS *)self)->mcts;
  mcts.~MCTS();
  Py_TYPE(self)->tp_free(self);
}

static PyObject *mcts_game_state(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;
  auto game_state = PythonHandle::copy(mcts.game_state().object);
  return game_state.steal();
}

static PyObject *mcts_expanded(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  if (mcts.expanded()) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyObject *mcts_complete(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  if (mcts.complete()) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyObject *mcts_collected(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  if (mcts.collected()) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyObject *mcts_turns(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;
  return PyLong_FromSize_t(mcts.turns());
}

static PyObject *mcts_add_dirichlet_noise(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char alpha_str[] = "alpha";
  static char *keyword_names[] = {alpha_str, NULL};

  double alpha;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", keyword_names, &alpha)) {
    return NULL;
  }

  auto &mcts = ((PyMCTS *)self)->mcts;

  try {
    mcts.add_dirichlet_noise(alpha);
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *mcts_select_leaf(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  auto leaf = mcts.select_leaf();

  if (leaf == NULL) {
    Py_RETURN_NONE;
  }

  PythonHandle capsule(PyCapsule_New((void *)leaf, "Node", NULL));

  if (capsule.null()) {
    return NULL;
  }

  auto game_state = PythonHandle::copy(leaf->state().object);

  PythonHandle tuple(PyTuple_New(2));

  if (tuple.null()) {
    return NULL;
  }

  PyTuple_SET_ITEM(tuple.object, 0, capsule.steal());
  PyTuple_SET_ITEM(tuple.object, 1, game_state.steal());

  return tuple.steal();
}

static PyObject *mcts_expand_leaf(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *leaf_capsule;
  double av;
  PyObject *expansion_sequence;

  static char leaf_str[] = "leaf";
  static char av_str[] = "av";
  static char expansion_str[] = "expansion";

  static char *keyword_names[] = {leaf_str, av_str, expansion_str, NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OdO", keyword_names, &leaf_capsule, &av, &expansion_sequence)) {
    return NULL;
  }

  if (!PyCapsule_CheckExact(leaf_capsule)) {
    PyErr_SetString(PyExc_TypeError, "bad leaf argument");
    return NULL;
  }

  auto leaf = (MCTS<PythonHandle, PythonHandle>::Node *)PyCapsule_GetPointer(leaf_capsule, "Node");

  if (leaf == NULL) {
    return NULL;
  }

  PythonHandle expansion_iter(PyObject_GetIter(expansion_sequence));

  if (expansion_iter.null()) {
    return NULL;
  }

  std::vector<MCTS<PythonHandle, PythonHandle>::ExpansionEntry> expansion;

  for (;;) {
    PythonHandle expansion_elem(PyIter_Next(expansion_iter.object));

    if (expansion_elem.null()) {
      break;
    }

    if (!assert_tuple_length(expansion_elem.object, 3)) {
      return NULL;
    }

    PyObject *move;
    PyObject *game_state;
    double prior_probability;

    if (!PyArg_ParseTuple(expansion_elem.object, "OOd", &move, &game_state, &prior_probability)) {
      return NULL;
    }

    MCTS<PythonHandle, PythonHandle>::ExpansionEntry expansion_entry = {
        PythonHandle::copy(move), PythonHandle::copy(game_state), prior_probability};

    expansion.emplace_back(std::move(expansion_entry));
  }

  if (PyErr_Occurred()) {
    return NULL;
  }

  auto &mcts = ((PyMCTS *)self)->mcts;

  try {
    mcts.expand_leaf(leaf, av, std::move(expansion));
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *mcts_move_greedy(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  if (!mcts.expanded()) {
    PyErr_SetString(PyExc_RuntimeError, "root node hasn't been expanded");
  }

  if (mcts.complete()) {
    PyErr_SetString(PyExc_RuntimeError, "game is over");
  }

  try {
    return PythonHandle::copy(mcts.move_greedy().object).steal();
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *mcts_move_proportional(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  if (!mcts.expanded()) {
    PyErr_SetString(PyExc_RuntimeError, "root node hasn't been expanded");
  }

  if (mcts.complete()) {
    PyErr_SetString(PyExc_RuntimeError, "game is over");
  }

  try {
    return PythonHandle::copy(mcts.move_proportional().object).steal();
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *mcts_collect_result(PyObject *self, PyObject *args) {
  (void)args;
  auto &mcts = ((PyMCTS *)self)->mcts;

  if (mcts.collected()) {
    PyErr_SetString(PyExc_RuntimeError, "results were already collected");
    return NULL;
  }

  std::pair<double, std::vector<MCTS<PythonHandle, PythonHandle>::HistoryEntry>> result;

  try {
    result = mcts.collect_result();
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  PythonHandle av = PyFloat_FromDouble(result.first);

  std::vector<MCTS<PythonHandle, PythonHandle>::HistoryEntry> &history = result.second;

  PythonHandle history_list = iterator_to_list(
      history.begin(), history.end(), [](MCTS<PythonHandle, PythonHandle>::HistoryEntry &entry) {
        const auto lambda = [](std::pair<PythonHandle, double> &move_and_probability) {
          PythonHandle tuple(
              Py_BuildValue("Nd", move_and_probability.first.object, move_and_probability.second));

          if (tuple.null()) {
            return PythonHandle(NULL);
          }

          move_and_probability.first.steal();

          return tuple;
        };

        PythonHandle search_probabilities = iterator_to_list(
            entry.search_probabilities.begin(), entry.search_probabilities.end(), lambda);

        PythonHandle tuple(PyTuple_New(2));

        if (tuple.null()) {
          return PythonHandle(NULL);
        }

        PyTuple_SET_ITEM(tuple.object, 0, entry.game_state.steal());
        PyTuple_SET_ITEM(tuple.object, 1, search_probabilities.steal());

        return tuple;
      });

  PythonHandle tuple(PyTuple_New(2));

  if (tuple.null()) {
    return NULL;
  }

  PyTuple_SET_ITEM(tuple.object, 0, av.steal());
  PyTuple_SET_ITEM(tuple.object, 1, history_list.steal());

  return tuple.steal();
}

static PyObject *mcts_reset(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char initial_state_str[] = "initial_state";
  static char *keyword_names[] = {initial_state_str, NULL};

  PyObject *initial_state;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keyword_names, &initial_state)) {
    return NULL;
  }

  auto &mcts = ((PyMCTS *)self)->mcts;

  try {
    mcts.reset(PythonHandle::copy(initial_state), PythonHandle::copy(Py_None));
  } catch (std::bad_alloc &) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_RETURN_NONE;
}

static bool assert_tuple_length(PyObject *tuple, size_t length) {
  if (PyTuple_Check(tuple) && (size_t)PyTuple_Size(tuple) == length) {
    return true;
  }

  PyErr_Format(PyExc_TypeError, "expected a tuple of length %u", (unsigned int)length);
  return false;
}
