import curses
import time
import logging
from model.dataset import DatasetName
from model.classifier import ClassifierName
from service.dataset_service import DatasetService
from service.classifier_service import ClassifierService
from service.analysis_service import AnalysisService

logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

dataset_service = DatasetService()
classifier_service = ClassifierService()
analysis_service = AnalysisService()

def normalize_dataset_name(dataset_name: str) -> DatasetName:
    dataset_mapping = {
        "Statlog (German Credit Data)": DatasetName.GERMAN,
        "Census Income": DatasetName.ADULT
    }
    return dataset_mapping.get(dataset_name)

def normalize_classifier_name(classifier_name: str) -> ClassifierName:
    classifier_mapping = {
        "XGBClassifier": ClassifierName.XGB,
        "Support Vector Classification (SVC)": ClassifierName.SVC,
        "Random Forest Classifier": ClassifierName.RFC,
        "Logistic Regression": ClassifierName.LR
    }
    return classifier_mapping.get(classifier_name)

def run_cli(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

    height, width = stdscr.getmaxyx()
    menu = ["Bias Mitigation Demo", "What-If Tool", "Exit"]
    current_option = 0

    selected_datasets = set()
    selected_classifiers = set()

    datasets_per_page = 5
    current_page = 0
    datasets_paginated_page = 0
    analysis_paginated_page = 0

    while True:
        stdscr.clear()

        # Main Menu Page
        if current_page == 0:
            for idx, option in enumerate(menu):
                if idx == current_option:
                    stdscr.addstr(idx, 0, f"> {option}", curses.A_REVERSE)
                else:
                    stdscr.addstr(idx, 0, f"  {option}")
            stdscr.refresh()

            key = stdscr.getch()
            if key == 27: # ESC to exit
                break
            elif key == 258: # down arrow
                current_option = (current_option + 1) % len(menu)
            elif key == 259: # up arrow
                current_option = (current_option - 1) % len(menu)
            elif key == 10: # enter key
                if menu[current_option] == "Bias Mitigation Demo":
                    current_page = 1 # move to dataset selection page
                elif menu[current_option] == "What-If Tool":
                    stdscr.addstr(0, 0, "What-If Tool selected")
                    stdscr.refresh()
                    stdscr.getch()
                elif menu[current_option] == "Exit":
                    stdscr.clear()
                    stdscr.addstr(height // 2, width // 2 - 10, "Exiting program...")
                    stdscr.refresh()
                    time.sleep(1)
                    break

        # Bias Mitigation Demo: Dataset Selection (Page 1)
        elif current_page == 1:
            datasets = dataset_service.get_datasets()
            current_dataset = 0

            while True:
                stdscr.clear()
                stdscr.addstr(0, 0, "Select datasets (press Enter to toggle, ESC to return)")

                start_idx = datasets_paginated_page * datasets_per_page
                end_idx = start_idx + datasets_per_page
                paginated_datasets = datasets[start_idx:end_idx]

                for idx, dataset in enumerate(paginated_datasets):
                    prefix = "[X]" if dataset.name in selected_datasets else "[ ]"
                    if idx == current_dataset:
                        stdscr.addstr(idx + 1, 0, f"> {prefix} {dataset.name}", curses.A_REVERSE)
                    else:
                        stdscr.addstr(idx + 1, 0, f"  {prefix} {dataset.name}")

                stdscr.addstr(len(paginated_datasets) + 2, 0, f"Page {datasets_paginated_page + 1}/{(len(datasets) // datasets_per_page) + 1}")
                stdscr.addstr(len(paginated_datasets) + 3, 0, "Press Left/Right arrow to navigate pages, Enter to toggle selection")

                dataset = paginated_datasets[current_dataset]
                stdscr.addstr(len(paginated_datasets) + 5, 0, f"Description: {dataset.description}")
                stdscr.addstr(len(paginated_datasets) + 6, 0, f"URL: {dataset.url}")
                stdscr.addstr(len(paginated_datasets) + 7, 0, f"Sensitive Features:")
                for idx, feature in enumerate(dataset.sensitive_features):
                    stdscr.addstr(len(paginated_datasets) + 8 + idx, 0, f"  {feature.name}: {feature.privileged} vs {feature.unprivileged}")

                gap = len(paginated_datasets) + 8 + len(dataset.sensitive_features)
                stdscr.addstr(gap + 1, 0, "[1] Go back to homepage   [2] Go to next step")

                stdscr.refresh()

                key = stdscr.getch()
                if key == 27: # ESC to go back to main menu
                    current_page = 0
                    break
                elif key == 258: # down arrow
                    current_dataset = (current_dataset + 1) % len(paginated_datasets)
                elif key == 259: # up arrow
                    current_dataset = (current_dataset - 1) % len(paginated_datasets)
                elif key == 10: # enter key to toggle selection
                    dataset_name = dataset.name
                    if dataset_name in selected_datasets:
                        selected_datasets.remove(dataset_name)
                    else:
                        selected_datasets.add(dataset_name)
                elif key == 261: # right arrow to go to next page
                    if datasets_paginated_page < (len(datasets) // datasets_per_page):
                        datasets_paginated_page += 1
                        current_dataset = 0
                elif key == 260: # left arrow to go to previous page
                    if datasets_paginated_page > 0:
                        datasets_paginated_page -= 1
                        current_dataset = 0
                elif key == ord('1'): # go back to homepage
                    current_page = 0
                    break
                elif key == ord('2'): # go to next step
                    if len(selected_datasets) == 0:
                        stdscr.addstr(len(paginated_datasets) + 10, 0, "You must select at least one dataset.")
                        stdscr.refresh()
                        stdscr.getch()
                    else:
                        current_page = 2 # move to classifier selection page
                        break

        # Bias Mitigation Demo: Classifier Selection (Page 2)
        elif current_page == 2:
            classifiers = classifier_service.get_classifiers()
            current_classifier = 0

            while True:
                stdscr.clear()
                stdscr.addstr(0, 0, "Select classifier (press Enter to toggle, ESC to return)")

                for idx, classifier in enumerate(classifiers):
                    prefix = "[X]" if classifier.name in selected_classifiers else "[ ]"
                    if idx == current_classifier:
                        stdscr.addstr(idx + 1, 0, f"> {prefix} {classifier.name}", curses.A_REVERSE)
                    else:
                        stdscr.addstr(idx + 1, 0, f"  {prefix} {classifier.name}")

                gap = len(classifiers) + 2
                stdscr.addstr(gap, 0, "[1] Go back to dataset selection   [2] Go to next step")

                stdscr.refresh()

                key = stdscr.getch()
                if key == 27: # ESC to go back to dataset selection
                    current_page = 1
                    break
                elif key == 258: # down arrow
                    current_classifier = (current_classifier + 1) % len(classifiers)
                elif key == 259: # up arrow
                    current_classifier = (current_classifier - 1) % len(classifiers)
                elif key == 10: # enter key to toggle classifier selection
                    classifier_name = classifiers[current_classifier].name
                    if classifier_name in selected_classifiers:
                        selected_classifiers.remove(classifier_name)
                    else:
                        selected_classifiers.add(classifier_name)
                elif key == ord('1'): # go back to dataset selection
                    current_page = 1
                    break
                elif key == ord('2'): # go to next step
                    current_page = 3
                    break

        # Bias Mitigation Demo: Analysis Results (Page 3)
        elif current_page == 3:
            selected_datasets_enum = [normalize_dataset_name(dataset) for dataset in selected_datasets]
            selected_classifiers_enum = [normalize_classifier_name(classifier) for classifier in selected_classifiers]
            # logging.debug(f"{selected_classifiers_enum}")

            analysis_result = analysis_service.analyse(selected_datasets_enum, selected_classifiers_enum)

            if not analysis_result:
                stdscr.addstr(1, 0, "No analysis results available.")
                stdscr.refresh()
                stdscr.getch()
                continue

            stdscr.clear()
            stdscr.addstr(0, 0, "Analysis Result:\n")
            column_widths = [10, 18, 18, 12, 18, 18, 12, 18, 12]
            headers = ["Dataset", "Classifier", "Sensitive Column", "Accuracy", "Statistical Parity", "Equal Opportunity", "Average Odds", "Disparate Impact", "Theil Index"]

            for idx, header in enumerate(headers):
                stdscr.addstr(1, sum(column_widths[:idx]), f"{header[:18]:<18}")

            total_results = len(analysis_result)
            max_rows_per_page = 5
            total_pages = (total_results // max_rows_per_page) + (1 if total_results % max_rows_per_page > 0 else 0)

            start_idx = analysis_paginated_page * max_rows_per_page
            end_idx = min(start_idx + max_rows_per_page, total_results)

            row_idx = 2
            for i in range(start_idx, end_idx):
                result = analysis_result[i]
                stdscr.addstr(row_idx, 0, f"{result['Dataset'][:18]:<18}")
                stdscr.addstr(row_idx, column_widths[0], f"{result['Classifier'][:18]:<18}")
                stdscr.addstr(row_idx, sum(column_widths[:2]), f"{result['Sensitive Column'][:18]:<18}")
                stdscr.addstr(row_idx, sum(column_widths[:3]), f"{result['Model Accuracy']:.5f}")
                stdscr.addstr(row_idx, sum(column_widths[:4]), f"{result['Statistical Parity Difference']:.5f}")
                stdscr.addstr(row_idx, sum(column_widths[:5]), f"{result['Equal Opportunity Difference']:.5f}")
                stdscr.addstr(row_idx, sum(column_widths[:6]), f"{result['Average Odds Difference']:.5f}")
                stdscr.addstr(row_idx, sum(column_widths[:7]), f"{result['Disparate Impact']:.5f}")
                stdscr.addstr(row_idx, sum(column_widths[:8]), f"{result['Theil Index']:.5f}")
                row_idx += 1

            stdscr.refresh()
            stdscr.addstr(row_idx + 1, 0, f"Page {analysis_paginated_page + 1}/{total_pages}")
            stdscr.addstr(row_idx + 2, 0, "Press Left/Right arrow to navigate pages.")

            stdscr.addstr(row_idx + 4, 0, "[1] Go to previous page   [2] Go to next page")

            stdscr.refresh()

            key = stdscr.getch()
            if key == ord('1'):
                current_page = 2
                stdscr.refresh()
            elif key == ord('2'):
                stdscr.clear()
                stdscr.refresh()
                stdscr.getch()
            elif key == 261: # right arrow
                if analysis_paginated_page < (total_results // 5):
                    analysis_paginated_page += 1
            elif key == 260: # left arrow
                if analysis_paginated_page > 0:
                    analysis_paginated_page -= 1
            elif key == 27: # ESC to return
                current_page = 2 # retun to classifier selection
                break

curses.wrapper(run_cli)
