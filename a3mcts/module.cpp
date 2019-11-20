#include <Python.h>
#include <stddef.h>

#include "mcts.h"

struct PythonHandle {
  PyObject *object;

  PythonHandle() : object(nullptr) {
  }

  PythonHandle(PyObject *object_) : object(object_) {
    Py_INCREF(object);
  }

  PythonHandle(const PythonHandle &) = delete;

  PythonHandle(PythonHandle &&other) : object(other.object) {
    other.object = nullptr;
  }

  PythonHandle &operator=(const PythonHandle &) = delete;

  PythonHandle &operator=(PythonHandle &&other) {
    if (object != nullptr) {
      Py_DECREF(object);
    }

    object = other.object;
    other.object = nullptr;

    return *this;
  }

  ~PythonHandle() {
    if (object != nullptr) {
      Py_DECREF(object);
    }
  }
};

struct TypeSpec {
  const char *name;
  size_t size;
  newfunc create;
  destructor destroy;
  PyMethodDef *methods;
  const char *docstring;
};

static PyObject *create_python_type(const TypeSpec *spec);

struct MCTSInstance {
  PyObject_HEAD
  MCTS<PythonHandle, PythonHandle> mcts;
};

static PyObject *
mcts_create(PyTypeObject *subtype, PyObject *args, PyObject *kwargs);

static void mcts_destroy(PyObject *self_);

static PyObject *mcts_game_state(PyObject *self, PyObject *args);
static PyObject *mcts_expanded(PyObject *self, PyObject *args);
static PyObject *mcts_select_leaf(PyObject *self, PyObject *args);
static PyObject* mcts_expand_leaf(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef mcts_methods[] = {
  { "game_state", mcts_game_state, METH_NOARGS, NULL },
  { "expanded", mcts_expanded, METH_NOARGS, NULL },
  { "select_leaf", mcts_select_leaf, METH_NOARGS, NULL },
  { "expand_leaf", (PyCFunction)mcts_expand_leaf, METH_VARARGS | METH_KEYWORDS, NULL },
  { NULL, NULL, -1, NULL }
};

const static TypeSpec mcts_typespec = {
    "MCTS", sizeof(MCTSInstance), mcts_create, mcts_destroy, mcts_methods, NULL};

struct LeafInstance {
  PyObject_HEAD
  MCTS<PythonHandle, PythonHandle>::Leaf leaf;
};

const static TypeSpec leaf_typespec = {
    "Leaf", sizeof(LeafInstance), NULL, NULL, NULL, NULL};  

static PyObject *get_leaf_type(void);

static PyModuleDef module_defn = {
    PyModuleDef_HEAD_INIT, "a3mcts", NULL, 0, NULL, NULL, NULL, NULL, NULL};

extern "C" {

PyObject *PyInit_a3mcts(void) {
  PyObject *module = PyModule_Create(&module_defn);

  if (module == NULL) {
    return NULL;
  }

  PyObject *mcts_type = create_python_type(&mcts_typespec);

  if(mcts_type == NULL) {
    Py_DECREF(module);
    return NULL;
  }

  if (PyModule_AddObject(module, mcts_typespec.name, mcts_type) < 0) {
    Py_DECREF(module);
    Py_DECREF(mcts_type);
    return NULL;
  }

  PyObject *leaf_type = create_python_type(&leaf_typespec);

  if(leaf_type == NULL) {
    Py_DECREF(module);
    return NULL;
  }

  if (PyModule_AddObject(module, leaf_typespec.name, leaf_type) < 0) {
    Py_DECREF(module);
    Py_DECREF(leaf_type);
    return NULL;
  }

  return module;
}
}

static PyObject *create_python_type(const TypeSpec *spec) {
  PyTypeObject *type = (PyTypeObject*)PyObject_Malloc(sizeof(PyTypeObject));

  if (type == NULL) {
    return NULL;
  }

  memset(type, 0, sizeof(PyTypeObject));
  PyObject_Init((PyObject *)type, &PyType_Type);

  type->tp_name = spec->name;
  type->tp_basicsize = spec->size;
  type->tp_new = spec->create;
  type->tp_dealloc = spec->destroy;
  type->tp_methods = spec->methods;
  type->tp_doc = spec->docstring;
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

  if (PyType_Ready(type) < 0) {
    Py_DECREF(type);
    return NULL;
  }

  return (PyObject*)type;
}

static PyObject *
mcts_create(PyTypeObject *subtype, PyObject *args, PyObject *kwargs) {
  PyObject *initial_state;
  double c_init;
  double c_base;

  static char keyword_names_str[] = "initial_state\000c_init\000c_base";

  static char *keyword_names[] = {
      keyword_names_str, keyword_names_str + 14, keyword_names_str + 21, NULL};

  if (!PyArg_ParseTupleAndKeywords(args,
                                   kwargs,
                                   "Odd",
                                   keyword_names,
                                   &initial_state,
                                   &c_init,
                                   &c_base)) {
    return NULL;
  }

  MCTSInstance *self = PyObject_New(MCTSInstance, subtype);

  if (self == NULL) {
    return NULL;
  }

  new (&self->mcts)
      MCTS<PythonHandle, PythonHandle>(c_init, c_base, initial_state);

  return (PyObject *)self;
}

static void mcts_destroy(PyObject *self_) {
  MCTSInstance *self = (MCTSInstance *)self_;
  self->mcts.~MCTS();
}

static PyObject *mcts_game_state(PyObject *self_, PyObject *args) {
  (void)args;

  MCTSInstance *self = (MCTSInstance*)self_;

  const PythonHandle &game_state = self->mcts.game_state();

  Py_INCREF(game_state.object);
  return game_state.object;
}

static PyObject *mcts_expanded(PyObject *self_, PyObject *args) {
  (void)args;

  MCTSInstance *self = (MCTSInstance*)self_;

  if(self->mcts.expanded()) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyObject *mcts_select_leaf(PyObject *self_, PyObject *args) {
  (void)args;

  MCTSInstance *self = (MCTSInstance*)self_;

  MCTS<PythonHandle, PythonHandle>::Leaf leaf = self->mcts.select_leaf();

  if(!leaf.present()) {
    Py_RETURN_NONE;
  }

  PyObject *leaf_type = get_leaf_type();

  if(leaf_type == NULL) {
    return NULL;
  }

  LeafInstance *leaf_instance = PyObject_New(LeafInstance, (PyTypeObject*)leaf_type);
  leaf_instance->leaf = leaf;

  return (PyObject*)leaf_instance;
}

static PyObject* mcts_expand_leaf(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *leaf_type = get_leaf_type();

  if(leaf_type == NULL) {
    return NULL;
  }

  PyObject *leaf;
  double av;
  PyObject *children;

  static char keyword_names_str[] = "leaf\000av\000children";

  static char *keyword_names[] = {
      keyword_names_str, keyword_names_str + 5, keyword_names_str + 8, NULL};

  if (!PyArg_ParseTupleAndKeywords(args,
                                   kwargs,
                                   "O!dO",
                                   keyword_names,
                                   leaf_type,
                                   &leaf,
                                   &av,
                                   &children)) {
    return NULL;
  }

  children = PyObject_GetIter(children);

  if(children == NULL) {
    return NULL;
  }

  std::vector<MCTS<PythonHandle, PythonHandle>::ExpansionEntry> expansion;

  for(;;) {
    PyObject *child = PyIter_Next(children);

    if(child == NULL) {
      break;
    }

    PyObject *move;
    PyObject *game_state;
    double prior_probability;

    // FIXME: handle non-tuples
    if(!PyArg_ParseTuple(child, "OOd", &move, &game_state, &prior_probability)) {
      return NULL;
    }

    MCTS<PythonHandle, PythonHandle>::ExpansionEntry entry = {
      PythonHandle(move),
      PythonHandle(game_state),
      prior_probability
    };

    expansion.push_back(std::move(entry));

    Py_DECREF(child);
  }

  Py_DECREF(children);

  Py_RETURN_NONE;
}

static PyObject *get_leaf_type(void) {
  PyObject *module = PyState_FindModule(&module_defn);

  PyObject *leaf_type = PyObject_GetAttrString(module, leaf_typespec.name);

  if(leaf_type == NULL) {
    return NULL;
  }

  if(PyObject_IsInstance(leaf_type, (PyObject*)&PyType_Type) != 1) {
    // FIXME: do something intelligent here
    return NULL;
  }

  return leaf_type;
}
