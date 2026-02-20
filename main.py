from download_data import main
from safety_guard import run_startup_safety_checks


if __name__ == "__main__":
    run_startup_safety_checks()
    main()
