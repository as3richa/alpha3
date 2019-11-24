#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

template <class GameState, class Move,
          class Generator = std::default_random_engine>
class MCTS {
public:
  struct Node {
  private:
    Move move;
    GameState game_state;

    double prior_probability;

    Node *parent;
    Node *child;
    Node *sibling;

    size_t n_visits;
    double total_av;

    bool expanded() const { return n_visits != 0; }

    bool terminal() const { return expanded() && child == nullptr; }

    friend class MCTS;

  public:
    const Move &prev_move() { return move; }

    const GameState &state() { return game_state; }
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

private:
  const double c_init;
  const double c_base;

  Node *root;
  Node *freelist;

  std::vector<HistoryEntry> history;

  size_t searches_this_turn_;

  Generator generator;

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

    free_subtree(subtree->child);
    free_node(subtree);
  }

  void ascend_tree(Node *node, double av) {
    while (node != nullptr) {
      node->n_visits++;
      node->total_av += av;
      node = node->parent;
      av = -av;
    }
  }

  const Move *play_move(Node *new_root) {
    const size_t denom = root->n_visits - 1;

    std::vector<std::pair<Move, double>> search_probabilities;

    size_t new_root_index = SIZE_MAX;

    for (Node *child = root->child; child != nullptr;) {
      search_probabilities.emplace_back(
          std::move(child->move),
          (denom == 0) ? 0.0 : ((double)child->n_visits / denom));

      Node *next = child->sibling;

      if (child == new_root) {
        new_root_index = search_probabilities.size() - 1;
      } else {
        free_subtree(child);
      }

      child = next;
    }

    HistoryEntry entry = {std::move(root->game_state),
                          std::move(search_probabilities)};
    history.emplace_back(std::move(entry));

    const Move *move = nullptr;

    if (new_root != nullptr) {
      new_root->move = std::move(root->move);
      new_root->parent = nullptr;
      new_root->sibling = nullptr;

      assert(new_root_index != SIZE_MAX);
      move = &history.back().search_probabilities[new_root_index].first;
    }

    free_node(root);
    root = new_root;

    searches_this_turn_ = 0;

    return move;
  }

public:
  MCTS(double c_init_, double c_base_, GameState initial_state = GameState(),
       Move phony_move = Move())
      : c_init(c_init_), c_base(c_base_), root(nullptr), freelist(nullptr),
        history(), generator(std::random_device{}()) {
    reset(std::move(initial_state), std::move(phony_move));
  }

  MCTS(MCTS &&other)
      : c_init(other.c_init), c_base(other.c_base), root(other.root),
        freelist(other.freelist), history(std::move(other.history)) {}

  ~MCTS() {
    free_subtree(root);

    for (Node *node = freelist; node != nullptr;) {
      Node *next = node->sibling;
      delete node;
      node = next;
    }
  }

  const GameState &game_state() const { return root->game_state; }

  bool expanded() const { return root != nullptr && root->expanded(); }

  bool complete() const {
    return root != nullptr && root->expanded() && root->terminal();
  }

  bool collected() const { return root == nullptr; }

  size_t turns() const { return history.size() + 1; }

  size_t searches_this_turn() const {
    assert(!collected());
    return searches_this_turn_;
  }

  void add_dirichlet_noise(double alpha, double fraction) {
    assert(expanded() && !complete());

    std::gamma_distribution<double> gamma(alpha, 1.0);

    std::vector<double> noise;
    double sum = 0.0;

    for (Node *child = root->child; child != nullptr; child = child->sibling) {
      const double value = gamma(generator);
      noise.push_back(value);
      sum += value;
    }

    for (auto &value : noise) {
      value /= sum;
    }

    auto it = noise.begin();

    for (Node *child = root->child; child != nullptr; child = child->sibling) {
      child->prior_probability = fraction * (*it) + (1 - fraction) * child->prior_probability;
      ++it;
    }
  }

  Node *select_leaf() {
    Node *node = root;

    while (node->expanded()) {
      if (node->terminal()) {
        node->n_visits++;
        ascend_tree(node->parent, -node->total_av);
        searches_this_turn_++;
        return nullptr;
      }

      Node *best_child = nullptr;
      double best_score = 0.0;

      for (Node *child = node->child; child != nullptr;
           child = child->sibling) {
        const double average_av =
            (child->n_visits == 0) ? 0.0 : (child->total_av / child->n_visits);

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
    return node;
  }

  void expand_leaf(Node *leaf, double av,
                   std::vector<ExpansionEntry> &&expansion) {
    assert(leaf != nullptr && !leaf->expanded());

    if (expansion.empty()) {
      leaf->child = nullptr;
    } else {
      Node *prev_child = nullptr;

      for (auto &&entry : expansion) {
        Node *child = alloc_node();

        child->move = std::move(entry.move);
        child->game_state = std::move(entry.game_state);
        child->prior_probability = entry.prior_probability;

        child->parent = leaf;
        child->child = nullptr;

        child->n_visits = 0;
        child->total_av = 0.0;

        if (prev_child == nullptr) {
          leaf->child = child;
        } else {
          prev_child->sibling = child;
        }

        prev_child = child;
      }

      prev_child->sibling = nullptr;
    }

    ascend_tree(leaf, av);

    searches_this_turn_++;
  }

  const Move &move_greedy() {
    assert(expanded() && !complete());

    Node *best = root->child;

    for (Node *child = best->sibling; child != nullptr;
         child = child->sibling) {
      if (child->n_visits > best->n_visits) {
        best = child;
      }
    }

    return *play_move(best);
  }

  const Move &move_proportional() {
    assert(expanded() && !complete());

    if (root->n_visits == 1) {
      size_t n_children = 1;
      Node *chosen = root->child;

      for (Node *child = chosen->sibling; child != nullptr;
           child = child->sibling) {
        std::uniform_int_distribution<size_t> distribution(0, n_children);

        if (distribution(generator) == 0) {
          chosen = child;
        }

        n_children++;
      }

      return *play_move(chosen);
    }

    std::uniform_int_distribution<size_t> distribution(0, root->n_visits - 2);
    size_t selector = distribution(generator);

    for (Node *child = root->child;; child = child->sibling) {
      assert(child != nullptr);

      if (selector < child->n_visits) {
        return *play_move(child);
      }

      selector -= child->n_visits;
    }

    assert(false);
  }

  std::pair<double, std::vector<HistoryEntry>> collect_result() {
    double score = root->terminal() ? root->total_av : 0.0;

    play_move(nullptr);

    if (history.size() % 2 == 0) {
      score *= -1;
    }

    auto result = std::make_pair(score, std::move(history));
    history.clear();

    assert(collected());

    return result;
  }

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

    history.clear();

    searches_this_turn_ = 0;
  }
};
