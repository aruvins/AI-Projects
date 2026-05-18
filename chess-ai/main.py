import sys

def main():
    print("\n♟️ Neural Chess AI (V1)")
    print("----------------------")
    print("1. Train model")
    print("2. Play vs AI")
    print("3. Exit\n")

    choice = input("Select option: ").strip()

    if choice == "1":
        from utils.train import train
        train()

    elif choice == "2":
        from utils.gui import ChessGUI

        gui = ChessGUI()
        gui.run()

    else:
        print("Exiting...")
        sys.exit()


if __name__ == "__main__":
    main()