#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

template <class GameState, class Move> class MCTS {
  struct Node {
    Move move;
    GameState game_state;

    double prior_probability;

    Node *parent;
    Node *child;
    Node *sibling;

    size_t n_visits;
    double total_av;

    bool expanded() const {
      return n_visits != 0;
    }

    bool terminal() const {
      return expanded() && child == nullptr;
    }
  };

public:
  struct Leaf {
  private:
    Node *node;

    Leaf(Node *node_) : node(node_) {
    }

    friend class MCTS;

  public:
    Leaf() : node(nullptr) {
    }

    bool present() const {
      return node != nullptr;
    }

    const GameState &game_state() const {
      assert(present());
      return node->game_state;
    }
  };

  struct ExpansionEntry {
    Move move;
    GameState game_state;
    double prior_probability;
  };

  struct HistoryEntry {
    GameState game_state;
    std::vector<std::pair<Move, double>> search_probabilities;
  };

  struct GameResult {
    double score;
    std::vector<HistoryEntry> history;
  };

private:
  const double c_init;
  const double c_base;

  Node *root;
  Node *freelist;

  std::vector<HistoryEntry> history;

  Node *alloc_node() {
    if (freelist != nullptr) {
      Node *node = freelist;
      freelist = node->sibling;
      return node;
    }

    return new Node;
  }

  void free_node(Node *node) {
    node->sibling = freelist;
    freelist = node;
  }

  void free_subtree(Node *subtree) {
    if (subtree == nullptr) {
      return;
    }

    free_subtree(subtree->sibling);
    free_subtree(subtree->child);
    free_node(subtree);
  }

  void ascend_tree(Node *node, double av) {
    while (node != root) {
      node->n_visits++;
      node->total_av += av;
      node = node->parent;
    }
  }

  void move(Node *new_root) {
    const size_t denom = root->n_visits - 1;

    std::vector<std::pair<Move, double>> search_probabilities;

    for (Node *child = root->child; child != nullptr; child = child->sibling) {
      search_probabilities.emplace_back(std::move(child->move),
                                        child->n_visits / denom);

      if (child != new_root) {
        free_subtree(child);
      }
    }

    new_root->move = std::move(root->move);
    new_root->parent = nullptr;

    history.emplace_back(std::move(new_root->move),
                         std::move(search_probabilities));

    free_node(root);
    root = new_root;
  }

public:
  MCTS(double c_init_,
       double c_base_,
       GameState initial_state = GameState(),
       Move phony_move = Move())
      : c_init(c_init_), c_base(c_base_), root(nullptr), freelist(nullptr),
        history() {
    reset(std::move(initial_state), std::move(phony_move));
  }

  MCTS(MCTS &&other)
      : c_init(other.c_init), c_base(other.c_base), root(other.root),
        freelist(other.freelist), history(std::move(other.history)) {
  }

  ~MCTS() {
    free_subtree(root);

    for (Node *node = freelist; node != nullptr;) {
      Node *next = node->sibling;
      delete node;
      node = next;
    }
  }

  const GameState &game_state() const {
    return root->game_state;
  }

  bool expanded() const {
    return root->expanded();
  }

  bool complete() const {
    assert(root->expanded());
    return root->terminal();
  }

  Leaf select_leaf() {
    Node *node = root;

    while (node->expanded()) {
      if (node->terminal()) {
        if (fabs(node->total_av) > 1e-7) {
          ascend_tree(node, node->total_av);
        }

        return Leaf(nullptr);
      }

      Node *best_child = nullptr;
      double best_score = 0.0;

      for (Node *child = node->child; child != nullptr;
           child = child->sibling) {
        const double average_av = child->total_av / child->n_visits;

        const double exploration =
            log((1 + node->n_visits + c_base) / c_base) + c_init;
        const double prior = child->prior_probability;
        const double u =
            exploration * prior * sqrt(node->n_visits) / (1 + child->n_visits);

        const double score = average_av + u;

        if (best_child == nullptr || score > best_score) {
          best_child = child;
          best_score = score;
        }
      }

      assert(best_child != nullptr);
      node = best_child;
    }

    assert(!node->expanded());
    return Leaf(node);
  }

  void
  expand_leaf(Leaf leaf, double av, std::vector<ExpansionEntry> &&expansion) {
    Node *node = leaf.node;
    assert(node != nullptr && !node->expanded());

    node->n_visits = 1;
    node->total_av = av;

    if (expansion.empty()) {
      node->child = nullptr;
    } else {
      Node *prev_child = nullptr;

      for (auto &&entry : expansion) {
        Node *child = alloc_node();

        child->move = std::move(entry.move);
        child->game_state = std::move(entry.game_state);
        child->prior_probability = entry.prior_probability;

        child->parent = node;
        child->child = nullptr;

        child->n_visits = 0;
        child->total_av = 0.0;

        if (prev_child == nullptr) {
          node->child = child;
        } else {
          prev_child->sibling = child;
        }
      }

      if (prev_child != nullptr) {
        prev_child->sibling = nullptr;
      }
    }

    ascend_tree(node, av);
  }

  void terminate_leaf(Leaf leaf, double score) {
    Node *node = leaf.node;
    assert(node != nullptr);
    assert(!node->expanded());

    node->n_visits = 1;
    node->total_av = score;

    ascend_tree(node, score);
  }

  const Move &move_greedy() {
    assert(root->expanded() && !root->terminal());

    Node *best = root->child;

    for (Node *child = best->sibling; child != nullptr;
         child = child->sibling) {
      if (child->n_visits > best->n_visits) {
        best = child;
      }
    }

    move(best);

    return best->move;
  }

  template <class Generator = std::default_random_engine>
  const Move &move_proportional(Generator &&generator = Generator()) {
    assert(root->expanded() && !root->terminal());

    std::uniform_int_distribution<size_t> distribution(0, root->n_visits - 2);
    size_t selector = distribution(generator);

    for (Node *child = root->child;; child = child->sibling) {
      assert(child != nullptr);

      if (selector < child->n_visits) {
        move(child);
        return child->move;
      }

      selector -= child->n_visits;
    }
  }

  /*GameResult finalize_result() {
    assert(root->terminal());

    move(nullptr);

    const double score = root->total_av;
    free_node(root);
    root = nullptr;

    //std::move(history)
    return GameResult{score};
  }*/

  void reset(GameState initial_state = GameState(), Move phony_move = Move()) {
    free_subtree(root);

    root = alloc_node();

    root->move = std::move(phony_move);
    root->game_state = std::move(initial_state);

    root->parent = nullptr;
    root->child = nullptr;
    root->sibling = nullptr;

    root->n_visits = 0;
    root->total_av = 0.0;

    // history.clear();
  }
};
