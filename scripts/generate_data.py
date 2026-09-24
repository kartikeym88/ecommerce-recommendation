import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.recsys.data.generator import SyntheticDataGenerator
from src.recsys.config import settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--products", type=int, default=200)
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    
    gen = SyntheticDataGenerator(
        n_users=args.users,
        n_products=args.products,
        n_days=args.days,
        seed=args.seed
    )
    
    print(f"Generating data for {args.users} users and {args.products} products over {args.days} days...")
    gen.write(settings.paths.data_dir)
    print("Done! Data saved to data/raw/")

if __name__ == "__main__":
    main()
