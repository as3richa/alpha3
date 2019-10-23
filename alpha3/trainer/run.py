from coordinator import Coordinator

def run():
    coordinator = Coordinator(4)
    coordinator.train(None, 13, 37)

if __name__ == "__main__":
    run()
