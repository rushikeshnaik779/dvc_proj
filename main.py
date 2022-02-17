import argparse
import src.cleaning
import src.train_test_model
import src.check_score

import logging


def run_pipeline(args):

    logging.basicConfig(level=logging.INFO)
    print("Cleaning started...")
    if args.action == "all" or args.action =="cleaning":
        logging.info("Cleaning started...")
        src.cleaning.execute_cleaning()


    if args.action == "all" or args.action == "train_test_model":
        logging.info("training and test splliting started ")
        #src.train_test_model.train_test_split()
        src.train_test_model.train_test_model()



    if args.action =="all" or args.action == "check_score":
        logging.info("Score check procedure started")
        src.check_score.check_score()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Training Pipeline"
    )

    parser.add_argument(
        "--action", 
        type=str,
        choices = ['cleaning', 'train_test_model', "check_score", "all"], 
        default="all", 
        help="pipeline action"
    )

    main_args = parser.parse_args()

    run_pipeline(main_args)
