#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

template <class GameState> class MCTS {
  struct Node {
    GameState game_state;
    double prior_probability;

    Node *parent;
    Node *child;
    Node *sibling;

    size_t n_visits;
    double total_av;
  };

  Node *root;
  Node *freelist;

  double c_init;
  double c_base;

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

  void ascend_tree(Node *node, double av) {
    while (node != root) {
      node->n_visits++;
      node->total_av += av;
      node = node->parent;
    }
  }

public:
  struct Leaf {
    Node *node;

  public:
    bool present() {
      return node != nullptr;
    }

    const GameState &game_state() {
      return node->game_state;
    }
  };

  struct GameResult {
    struct Position {
      size_t move;
      GameState game_state;
      std::vector<std::pair<size_t, double>> search_probabilities;
    };

    double score;
    std::vector<Position> positions;
  };

  MCTS(double c_init_, double c_base_, GameState initial_state)
      : root(new Node{initial_state, 0.0, nullptr, nullptr, nullptr, 0, 0.0}), freelist(nullptr),
        c_init(c_init_), c_base(c_base_) {
  }

  Leaf select_leaf() {
    Node *node = root;

    while (node->n_visits > 0) {
      if (node->child == nullptr) {
        if (fabs(node->total_av) > 1e-7) {
          ascend_tree(node, node->total_av);
        }

        return Leaf{nullptr};
      }

      Node *best_child = nullptr;
      double best_score = 0.0;

      for (Node *child = node->child; child != nullptr; child = child->sibling) {
        const double average_av = child->total_av / child->n_visits;

        const double exploration = log((1 + node->n_visits + c_base) / c_base) + c_init;
        const double prior = child->prior_probability;
        const double u = exploration * prior * sqrt(node->n_visits) / (1 + child->n_visits);

        const double score = average_av + u;

        if (best_child == nullptr || score > best_score) {
          best_child = child;
          best_score = score;
        }
      }

      assert(best_child != nullptr);
      node = best_child;
    }

    assert(node != nullptr);
    assert(node->n_visits == 0);

    return Leaf{node};
  }

  void expand_leaf(Leaf leaf, double av, std::vector<std::pair<GameState, double>> children) {
    Node *node = leaf.node;
    assert(node != nullptr);
    assert(node->n_visits == 0);

    Node *prev_child = nullptr;

    for (const auto &game_state_and_prior : children) {
      Node *child = alloc_node();

      child->game_state = std::move(game_state_and_prior.first);
      child->prior_probability = game_state_and_prior.second;

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

    ascend_tree(node, av);
  }

  void terminate_leaf(Leaf leaf, double score) {
    Node *node = leaf.node;
    assert(node != nullptr);
    assert(node->n_visits == 0);

    node->n_visits = 1;
    node->total_av = score;

    ascend_tree(node, score);
  }

  bool move_greedily() {
    return false;
  }

  bool move_proportionally() {
    return false;
  }

  GameResult result() {
    return GameResult{0.0, {}};
  }
};

struct GameState {};

int main(void) {
  MCTS<GameState> mcts(0.0, 0.0, GameState{});
  MCTS<GameState>::Leaf leaf = mcts.select_leaf();
  mcts.expand_leaf(leaf, 0.0, std::vector<std::pair<GameState, double>>());
  mcts.terminate_leaf(leaf, -1);
  mcts.move_greedily();
  mcts.move_proportionally();
  mcts.result();
  return 0;
}
