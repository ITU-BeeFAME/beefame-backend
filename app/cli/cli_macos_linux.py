import curses
import time
import logging
from typing import Optional
from model.dataset import DatasetName
from model.classifier import ClassifierName
from model.method import MethodName
from service.dataset_service import DatasetService
from service.classifier_service import ClassifierService
from service.method_service import MethodService
from service.analysis_service import AnalysisService
from service.evaluation_service import EvaluationService

# logging to write file only
logging.basicConfig(
    filename='debug.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# suppress loggers
loggers_to_suppress = [
    'sklearn', 'xgboost', 'methods.preprocessing', 'ucimlrepo',
    'pandas', 'numpy', 'aequitas', 'methods', 'preprocessing',
    'methods.preprocessing.PrevalenceSampling'
]

for logger_name in loggers_to_suppress:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    logger.handlers = []
logging.getLogger().propagate = False

dataset_service = DatasetService()
classifier_service = ClassifierService()
method_service = MethodService()
analysis_service = AnalysisService()
evaluation_service = EvaluationService()

def normalize_dataset_name(dataset_name: str) -> Optional[DatasetName]:
    try:
        dataset_mapping = {
            "Statlog (German Credit Data)": DatasetName.GERMAN,
            "Census Income": DatasetName.ADULT
        }
        result = dataset_mapping.get(dataset_name)
        if result is None:
            logger.warning(f"Unknown dataset name: {dataset_name}")
        return result
    except Exception as e:
        logger.error(f"Error normalizing dataset name: {e}")
        return None

def normalize_classifier_name(classifier_name: str) -> Optional[ClassifierName]:
    try:
        classifier_mapping = {
            "XGBClassifier": ClassifierName.XGB,
            "Support Vector Classification (SVC)": ClassifierName.SVC,
            "Random Forest Classifier": ClassifierName.RFC,
            "Logistic Regression": ClassifierName.LR
        }
        result = classifier_mapping.get(classifier_name)
        if result is None:
            logger.warning(f"Unknown classifier name: {classifier_name}")
        return result
    except Exception as e:
        logger.error(f"Error normalizing classifier name: {e}")
        return None

def show_error_message(stdscr, message: str, duration: int = 2):
    """Display an error message on the screen for a specified duration."""
    height, width = stdscr.getmaxyx()
    stdscr.addstr(height - 1, 0, f"Error: {message}", curses.A_REVERSE)
    stdscr.refresh()
    time.sleep(duration)

def draw_menu(stdscr, menu: list, current_option: int):
    """Draw the main menu with the current option highlighted."""
    for idx, option in enumerate(menu):
        if idx == current_option:
            stdscr.addstr(idx, 0, f"> {option}", curses.A_REVERSE)
        else:
            stdscr.addstr(idx, 0, f"  {option}")
    stdscr.refresh()

def handle_main_menu(stdscr, current_option: int, height: int, width: int) -> tuple[int, bool, str]:
    """Handle main menu navigation and selection."""
    menu = ["Bias Mitigation Demo", "What-If Tool", "Exit"]
    key = stdscr.getch()
    
    if key == 27:  # ESC to exit
        return current_option, True, ""
    elif key == 258:  # down arrow
        return (current_option + 1) % len(menu), False, ""
    elif key == 259:  # up arrow
        return (current_option - 1) % len(menu), False, ""
    elif key == 10:  # enter key
        if menu[current_option] == "Bias Mitigation Demo":
            return current_option, False, "Bias Mitigation Demo"
        elif menu[current_option] == "What-If Tool":
            stdscr.addstr(0, 0, "What-If Tool selected")
            stdscr.refresh()
            stdscr.getch()
            return current_option, False, ""
        elif menu[current_option] == "Exit":
            stdscr.clear()
            stdscr.addstr(height // 2, width // 2 - 10, "Exiting program...")
            stdscr.refresh()
            time.sleep(1)
            return current_option, True, ""
    return current_option, False, ""

def draw_dataset_selection(stdscr, datasets: list, current_dataset: int, selected_datasets: set,
                         datasets_paginated_page: int, datasets_per_page: int):
    """Draw the dataset selection screen."""
    # clear if this is the first time drawing this page
    if not hasattr(draw_dataset_selection, 'last_page') or draw_dataset_selection.last_page != datasets_paginated_page:
        stdscr.clear()
        draw_dataset_selection.last_page = datasets_paginated_page

    stdscr.addstr(0, 0, "Select datasets (press Enter to toggle, ESC to return)")

    start_idx = datasets_paginated_page * datasets_per_page
    end_idx = start_idx + datasets_per_page
    paginated_datasets = datasets[start_idx:end_idx]

    # draw current selections
    for idx, dataset in enumerate(paginated_datasets):
        prefix = "[X]" if dataset.name in selected_datasets else "[ ]"
        if idx == current_dataset:
            stdscr.addstr(idx + 1, 0, f"> {prefix} {dataset.name}", curses.A_REVERSE)
        else:
            stdscr.addstr(idx + 1, 0, f"  {prefix} {dataset.name}")

    # clear and redraw pagination info
    pagination_start = len(paginated_datasets) + 2
    stdscr.move(pagination_start, 0)
    stdscr.clrtoeol()
    stdscr.addstr(pagination_start, 0, f"Page {datasets_paginated_page + 1}/{(len(datasets) // datasets_per_page) + 1}")
    
    stdscr.move(pagination_start + 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(pagination_start + 1, 0, "Press Left/Right arrow to navigate pages, Enter to toggle selection")

    # clear the description area first
    description_start_line = pagination_start + 3
    stdscr.move(description_start_line, 0)
    stdscr.clrtoeol()
    
    # draw dataset description
    dataset = paginated_datasets[current_dataset]
    stdscr.addstr(description_start_line, 0, f"Description: {dataset.description}")
    
    # clear URL line
    stdscr.move(description_start_line + 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(description_start_line + 1, 0, f"URL: {dataset.url}")
    
    # clear sensitive features area
    stdscr.move(description_start_line + 2, 0)
    stdscr.clrtoeol()
    stdscr.addstr(description_start_line + 2, 0, f"Sensitive Features:")
    
    # clear and redraw sensitive features
    for idx, feature in enumerate(dataset.sensitive_features):
        stdscr.move(description_start_line + 3 + idx, 0)
        stdscr.clrtoeol()
        stdscr.addstr(description_start_line + 3 + idx, 0, 
                     f"  {feature.name}: {feature.privileged} vs {feature.unprivileged}")

    # clear any remaining lines from previous dataset's sensitive features
    max_previous_features = 3  # maximum number of sensitive features possible
    for idx in range(len(dataset.sensitive_features), max_previous_features):
        stdscr.move(description_start_line + 3 + idx, 0)
        stdscr.clrtoeol()

    # clear and redraw navigation options
    nav_start_line = description_start_line + 3 + max_previous_features
    stdscr.move(nav_start_line, 0)
    stdscr.clrtoeol()
    stdscr.addstr(nav_start_line, 0, "[1] Go back to homepage   [2] Go to next step")
    
    # only refresh once at the end
    stdscr.refresh()

def handle_dataset_selection(stdscr, datasets: list, current_dataset: int, selected_datasets: set,
                           datasets_paginated_page: int, datasets_per_page: int) -> tuple[int, int, int, set]:
    """Handle dataset selection navigation and selection."""
    key = stdscr.getch()
    
    if key == 27:  # ESC to go back to main menu
        return 0, current_dataset, datasets_paginated_page, selected_datasets
    elif key == 258:  # down arrow
        return 1, (current_dataset + 1) % len(datasets), datasets_paginated_page, selected_datasets
    elif key == 259:  # up arrow
        return 1, (current_dataset - 1) % len(datasets), datasets_paginated_page, selected_datasets
    elif key == 10:  # enter key to toggle selection
        dataset_name = datasets[current_dataset].name
        if dataset_name in selected_datasets:
            selected_datasets.remove(dataset_name)
        else:
            selected_datasets.add(dataset_name)
        return 1, current_dataset, datasets_paginated_page, selected_datasets
    elif key == 261:  # right arrow
        if datasets_paginated_page < (len(datasets) // datasets_per_page):
            return 1, 0, datasets_paginated_page + 1, selected_datasets
    elif key == 260:  # left arrow
        if datasets_paginated_page > 0:
            return 1, 0, datasets_paginated_page - 1, selected_datasets
    elif key == ord('1'):  # go back to homepage
        return 0, current_dataset, datasets_paginated_page, selected_datasets
    elif key == ord('2'):  # go to next step
        if len(selected_datasets) == 0:
            show_error_message(stdscr, "You must select at least one dataset.")
        else:
            return 2, current_dataset, datasets_paginated_page, selected_datasets
    
    return 1, current_dataset, datasets_paginated_page, selected_datasets

def draw_classifier_selection(stdscr, classifiers: list, current_classifier: int, selected_classifiers: set):
    """Draw the classifier selection screen."""
    # clear if this is the first time drawing this page
    if not hasattr(draw_classifier_selection, 'last_page'):
        stdscr.clear()
        draw_classifier_selection.last_page = True

    stdscr.addstr(0, 0, "Select classifier (press Enter to toggle, ESC to return)")

    # draw method list
    for idx, classifier in enumerate(classifiers):
        prefix = "[X]" if classifier.name in selected_classifiers else "[ ]"
        if idx == current_classifier:
            # draw the checkbox without highlight
            stdscr.addstr(idx + 1, 0, prefix)
            # draw the classifier name with highlight
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(idx + 1, len(prefix) + 1, classifier.name)
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(idx + 1, 0, f"{prefix} {classifier.name}")

    # add navigation options
    gap = len(classifiers) + 2
    stdscr.addstr(gap, 0, "[1] Go back to dataset selection   [2] Go to next step")
    stdscr.refresh()

def handle_classifier_selection(stdscr, classifiers: list, current_classifier: int, 
                              selected_classifiers: set) -> tuple[int, int, set]:
    """Handle classifier selection navigation and selection."""
    key = stdscr.getch()
    
    if key == 27:  # ESC to go back to dataset selection
        return 1, current_classifier, selected_classifiers
    elif key == 258:  # down arrow
        return 2, (current_classifier + 1) % len(classifiers), selected_classifiers
    elif key == 259:  # up arrow
        return 2, (current_classifier - 1) % len(classifiers), selected_classifiers
    elif key == 10:  # enter key to toggle classifier selection
        classifier_name = classifiers[current_classifier].name
        if classifier_name in selected_classifiers:
            selected_classifiers.remove(classifier_name)
        else:
            selected_classifiers.add(classifier_name)
        return 2, current_classifier, selected_classifiers
    elif key == ord('1'):  # go back to dataset selection
        return 1, current_classifier, selected_classifiers
    elif key == ord('2'):  # go to next step
        if len(selected_classifiers) == 0:
            show_error_message(stdscr, "You must select at least one classifier.")
        else:
            return 3, current_classifier, selected_classifiers
    
    return 2, current_classifier, selected_classifiers

def draw_analysis_results(stdscr, analysis_result: list, analysis_paginated_page: int, 
                         max_rows_per_page: int = 5):
    """Draw the analysis results screen."""
    if not analysis_result:
        stdscr.addstr(1, 0, "No analysis results available.")
        stdscr.refresh()
        stdscr.getch()
        return

    # cache columns if not already cached
    if not hasattr(draw_analysis_results, 'column_config'):
        draw_analysis_results.column_config = {
            "Dataset": 15,
            "Classifier": 20,
            "Sensitive Column": 20,
            "Accuracy": 15,
            "Statistical Parity": 20,
            "Equal Opportunity": 20,
            "Average Odds": 15,
            "Disparate Impact": 20,
            "Theil Index": 15
        }
        
        current_position = 0
        draw_analysis_results.column_positions = {}
        for header, width in draw_analysis_results.column_config.items():
            draw_analysis_results.column_positions[header] = current_position
            current_position += width
        
        draw_analysis_results.headers = list(draw_analysis_results.column_config.keys())

    # clear and redraw headers if page changed
    if not hasattr(draw_analysis_results, 'last_page') or draw_analysis_results.last_page != analysis_paginated_page:
        stdscr.clear()
        stdscr.addstr(0, 0, "Analysis Result:\n")
        
        # draw headers
        for header in draw_analysis_results.headers:
            pos = draw_analysis_results.column_positions[header]
            width = draw_analysis_results.column_config[header]
            stdscr.addstr(1, pos, f"{header:<{width}}")
        
        draw_analysis_results.last_page = analysis_paginated_page

    # calculate pagination
    total_results = len(analysis_result)
    total_pages = (total_results + max_rows_per_page - 1) // max_rows_per_page
    current_page = min(analysis_paginated_page, total_pages - 1)

    # calculate visible range
    start_idx = current_page * max_rows_per_page
    end_idx = min(start_idx + max_rows_per_page, total_results)

    # clear only the results area
    for i in range(2, 2 + max_rows_per_page):
        stdscr.move(i, 0)
        stdscr.clrtoeol()

    # draw results
    row_idx = 2
    for i in range(start_idx, end_idx):
        result = analysis_result[i]
        
        # draw the result using column positions
        for header in draw_analysis_results.headers:
            pos = draw_analysis_results.column_positions[header]
            width = draw_analysis_results.column_config[header]
            
            if header == "Dataset":
                value = result['Dataset']
            elif header == "Classifier":
                value = result['Classifier']
            elif header == "Sensitive Column":
                value = result['Sensitive Column']
            elif header == "Accuracy":
                value = f"{result['Model Accuracy']:.3f}"
            elif header == "Statistical Parity":
                value = f"{result['Statistical Parity Difference']:.3f}"
            elif header == "Equal Opportunity":
                value = f"{result['Equal Opportunity Difference']:.3f}"
            elif header == "Average Odds":
                value = f"{result['Average Odds Difference']:.3f}"
            elif header == "Disparate Impact":
                value = f"{result['Disparate Impact']:.3f}"
            elif header == "Theil Index":
                value = f"{result['Theil Index']:.3f}"
                
            stdscr.addstr(row_idx, pos, f"{value:<{width}}")
        row_idx += 1

    # update pagination info
    info_row = 2 + max_rows_per_page
    stdscr.move(info_row, 0)
    stdscr.clrtoeol()
    stdscr.addstr(info_row, 0, f"Page {current_page + 1}/{total_pages}")
    stdscr.move(info_row + 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(info_row + 1, 0, "Press Left/Right arrow to navigate pages.")
    stdscr.move(info_row + 3, 0)
    stdscr.clrtoeol()
    stdscr.addstr(info_row + 3, 0, "[1] Go to previous page   [2] Go to next page")
    stdscr.refresh()

def handle_analysis_results(stdscr, analysis_result: list, analysis_paginated_page: int, 
                          total_results: int, max_rows_per_page: int = 5) -> tuple[int, int]:
    """Handle analysis results navigation."""
    key = stdscr.getch()
    
    # calculate total pages once
    total_pages = (total_results + max_rows_per_page - 1) // max_rows_per_page
    
    if key == ord('1'):  # go back to classifier selection
        return 2, analysis_paginated_page
    elif key == ord('2'):  # go to method selection
        return 4, analysis_paginated_page
    elif key == 261:  # right arrow
        if analysis_paginated_page < total_pages - 1:  # allow navigation up to last page
            return 3, analysis_paginated_page + 1
    elif key == 260:  # left arrow
        if analysis_paginated_page > 0:
            return 3, analysis_paginated_page - 1
    elif key == 27:  # ESC to return
        return 2, analysis_paginated_page
    
    return 3, analysis_paginated_page

def draw_method_selection(stdscr, methods: list, current_method: int, selected_methods: set):
    """Draw the method selection screen."""
    # clear if this is the first time drawing this page
    if not hasattr(draw_method_selection, 'last_page'):
        stdscr.clear()
        draw_method_selection.last_page = True

    stdscr.addstr(0, 0, "Select fairness mitigation methods (press Enter to toggle, ESC to return)")

    # draw method list
    for idx, method in enumerate(methods):
        prefix = "[X]" if method.name in selected_methods else "[ ]"
        if idx == current_method:
            # draw the checkbox without highlight
            stdscr.addstr(idx + 1, 0, prefix)
            # draw the method name with highlight
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(idx + 1, len(prefix) + 1, method.name)
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(idx + 1, 0, f"{prefix} {method.name}")

    # clear the description area first
    description_start_line = len(methods) + 2
    stdscr.move(description_start_line, 0)
    stdscr.clrtoeol()
    
    # show description for selected method after all options
    if methods:
        selected_method = methods[current_method]
        stdscr.addstr(description_start_line, 0, f"Description: {selected_method.description}")

    # add navigation options after description
    gap = len(methods) + 4
    stdscr.addstr(gap, 0, "[1] Go back to analysis results   [2] Go to next step")
    stdscr.refresh()

def handle_method_selection(stdscr, methods: list, current_method: int, 
                          selected_methods: set) -> tuple[int, int, set]:
    """Handle method selection navigation and selection."""
    key = stdscr.getch()
    
    if key == 27:  # ESC to go back to analysis results
        return 3, current_method, selected_methods
    elif key == 258:  # down arrow
        return 4, (current_method + 1) % len(methods), selected_methods
    elif key == 259:  # up arrow
        return 4, (current_method - 1) % len(methods), selected_methods
    elif key == 10:  # enter key to toggle method selection
        method_name = methods[current_method].name
        if method_name in selected_methods:
            selected_methods.remove(method_name)
        else:
            selected_methods.add(method_name)
        # reset evaluation state when methods are changed
        run_cli.evaluation_done = False
        run_cli.evaluation_result = None
        return 4, current_method, selected_methods
    elif key == ord('1'):  # go back to analysis results
        return 3, current_method, selected_methods
    elif key == ord('2'):  # go to next step
        if len(selected_methods) == 0:
            show_error_message(stdscr, "You must select at least one method.")
        else:
            # reset evaluation state before evaluation
            run_cli.evaluation_done = False
            run_cli.evaluation_result = None
            return 5, current_method, selected_methods
    
    return 4, current_method, selected_methods

def draw_evaluation_results(stdscr, evaluation_result: list, evaluation_paginated_page: int, 
                          max_rows_per_page: int = 5):
    """Draw the evaluation results screen."""
    if not evaluation_result:
        stdscr.addstr(1, 0, "No evaluation results available.")
        stdscr.refresh()
        stdscr.getch()
        return

    if not hasattr(draw_evaluation_results, 'last_page') or draw_evaluation_results.last_page != evaluation_paginated_page:
        stdscr.clear()
        draw_evaluation_results.last_page = evaluation_paginated_page

    stdscr.addstr(0, 0, "Evaluation Results:\n")
    
    # pre calculate column positions for better performance
    column_widths = [15, 20, 20, 15, 20, 20, 15, 20, 15]
    column_positions = [sum(column_widths[:i]) for i in range(len(column_widths))]
    
    headers = ["Dataset", "Classifier", "Method", "Accuracy", 
              "Statistical Parity", "Equal Opportunity", "Average Odds", 
              "Disparate Impact", "Theil Index"]

    # draw headers
    for idx, header in enumerate(headers):
        stdscr.addstr(1, column_positions[idx], f"{header:<20}")

    # calculate pagination
    total_results = len(evaluation_result)
    total_pages = (total_results + max_rows_per_page - 1) // max_rows_per_page
    current_page = min(evaluation_paginated_page, total_pages - 1)

    # calculate visible range
    start_idx = current_page * max_rows_per_page
    end_idx = min(start_idx + max_rows_per_page, total_results)

    # draw results
    row_idx = 2
    for i in range(start_idx, end_idx):
        result = evaluation_result[i]
        
        # find corresponding analysis result
        analysis_result = None
        for analysis in run_cli.analysis_result:
            if (analysis['Dataset'] == result['Dataset Name'] and 
                analysis['Classifier'] == result['Model Name'] and 
                analysis['Sensitive Column'] == result['Sensitive Column']):
                analysis_result = analysis
                break
        
        # calculate differences if analysis result exists
        accuracy_diff = result['Model Accuracy'] - analysis_result['Model Accuracy'] if analysis_result else 0
        stat_parity_diff = result['Statistical Parity Difference'] - analysis_result['Statistical Parity Difference'] if analysis_result else 0
        equal_opp_diff = result['Equal Opportunity Difference'] - analysis_result['Equal Opportunity Difference'] if analysis_result else 0
        avg_odds_diff = result['Average Odds Difference'] - analysis_result['Average Odds Difference'] if analysis_result else 0
        disparate_impact_diff = result['Disparate Impact'] - analysis_result['Disparate Impact'] if analysis_result else 0
        theil_index_diff = result['Theil Index'] - analysis_result['Theil Index'] if analysis_result else 0
        
        # format differences with signs
        def format_diff(value):
            sign = '+' if value > 0 else ''
            return f"{sign}{value:.3f}"
        
        # clear the result lines first
        stdscr.move(row_idx, 0)
        stdscr.clrtoeol()
        stdscr.move(row_idx + 1, 0)
        stdscr.clrtoeol()
        
        # draw the result with differences
        stdscr.addstr(row_idx, column_positions[0], f"{result['Dataset Name']:<15}")
        stdscr.addstr(row_idx, column_positions[1], f"{result['Model Name']:<20}")
        stdscr.addstr(row_idx, column_positions[2], f"{result['Method Name']:<20}")
        stdscr.addstr(row_idx, column_positions[3], f"{result['Model Accuracy']:.3f} {format_diff(accuracy_diff):<8}")
        stdscr.addstr(row_idx, column_positions[4], f"{result['Statistical Parity Difference']:.3f} {format_diff(stat_parity_diff):<8}")
        stdscr.addstr(row_idx, column_positions[5], f"{result['Equal Opportunity Difference']:.3f} {format_diff(equal_opp_diff):<8}")
        stdscr.addstr(row_idx, column_positions[6], f"{result['Average Odds Difference']:.3f} {format_diff(avg_odds_diff):<8}")
        stdscr.addstr(row_idx, column_positions[7], f"{result['Disparate Impact']:.3f} {format_diff(disparate_impact_diff):<8}")
        stdscr.addstr(row_idx, column_positions[8], f"{result['Theil Index']:.3f} {format_diff(theil_index_diff):<8}")
        
        # add sensitive attribute information
        stdscr.addstr(row_idx + 1, 0, f"Sensitive Attribute: {result['Sensitive Column']}")
        row_idx += 2

    # clear any remaining lines from previous page
    while row_idx < 2 + max_rows_per_page * 2:
        stdscr.move(row_idx, 0)
        stdscr.clrtoeol()
        row_idx += 1

    # draw pagination info
    stdscr.addstr(row_idx + 1, 0, f"Page {current_page + 1}/{total_pages}")
    stdscr.addstr(row_idx + 2, 0, "Press Left/Right arrow to navigate pages.")
    stdscr.addstr(row_idx + 4, 0, "[1] Go back to method selection   [2] Go to next page")
    stdscr.refresh()

def handle_evaluation_results(stdscr, evaluation_result: list, evaluation_paginated_page: int, 
                            total_results: int, max_rows_per_page: int = 5) -> tuple[int, int]:
    """Handle evaluation results navigation."""
    key = stdscr.getch()
    
    # calculate total pages once
    total_pages = (total_results + max_rows_per_page - 1) // max_rows_per_page
    
    if key == ord('1'):  # go back to method selection
        return 4, evaluation_paginated_page
    elif key == ord('2'):  # go to next page
        return 5, evaluation_paginated_page
    elif key == 261:  # right arrow
        if evaluation_paginated_page < total_pages - 1:  # allow navigation up to last page
            return 5, evaluation_paginated_page + 1
    elif key == 260:  # left arrow
        if evaluation_paginated_page > 0:
            return 5, evaluation_paginated_page - 1
    elif key == 27:  # ESC to return
        return 4, evaluation_paginated_page
    
    return 5, evaluation_paginated_page

def run_cli(stdscr):
    """Main CLI entry point."""
    try:
        curses.curs_set(0)
        stdscr.nodelay(1)
        curses.start_color()
        curses.use_default_colors()
        
        height, width = stdscr.getmaxyx()
        if height < 24 or width < 80:
            raise curses.error("Terminal window too small. Minimum size: 80x24")

        menu = ["Bias Mitigation Demo", "What-If Tool", "Exit"]
        current_option = 0
        current_page = 0
        previous_page = -1  # track previous page to know when to clear screen

        selected_datasets = set()
        selected_classifiers = set()
        selected_methods = set()

        datasets_per_page = 5
        datasets_paginated_page = 0
        analysis_paginated_page = 0
        evaluation_paginated_page = 0
        current_dataset = 0
        current_classifier = 0
        current_method = 0

        # initialize storage for results
        run_cli.analysis_result = None
        run_cli.evaluation_result = None
        run_cli.evaluation_done = False

        while True:
            try:
                # clear screen when changing pages
                if current_page != previous_page:
                    stdscr.clear()
                    previous_page = current_page

                if current_page == 0:
                    draw_menu(stdscr, menu, current_option)
                    current_option, should_exit, selected_option = handle_main_menu(stdscr, current_option, height, width)
                    if should_exit:
                        break
                    if selected_option == "Bias Mitigation Demo":
                        current_page = 1
                    elif selected_option == "What-If Tool":
                        current_page = 0
                    elif selected_option == "Exit":
                        break

                elif current_page == 1:
                    datasets = dataset_service.get_datasets()
                    draw_dataset_selection(stdscr, datasets, current_dataset, selected_datasets,
                                        datasets_paginated_page, datasets_per_page)
                    current_page, current_dataset, datasets_paginated_page, selected_datasets = handle_dataset_selection(stdscr, datasets, current_dataset, selected_datasets,
                                              datasets_paginated_page, datasets_per_page)

                elif current_page == 2:
                    classifiers = classifier_service.get_classifiers()
                    draw_classifier_selection(stdscr, classifiers, current_classifier, selected_classifiers)
                    current_page, current_classifier, selected_classifiers = handle_classifier_selection(stdscr, classifiers, current_classifier, selected_classifiers)

                elif current_page == 3:
                    selected_datasets_enum = [normalize_dataset_name(dataset) for dataset in selected_datasets]
                    selected_classifiers_enum = [normalize_classifier_name(classifier) for classifier in selected_classifiers]
                    
                    try:
                        run_cli.analysis_result = analysis_service.analyse(selected_datasets_enum, selected_classifiers_enum)
                        draw_analysis_results(stdscr, run_cli.analysis_result, analysis_paginated_page)
                        current_page, analysis_paginated_page = handle_analysis_results(stdscr, run_cli.analysis_result, analysis_paginated_page, len(run_cli.analysis_result))
                    except Exception as e:
                        logger.error(f"Error during analysis: {e}")
                        show_error_message(stdscr, "Error during analysis. Check debug.log for details.")
                        time.sleep(2)
                        current_page = 2

                elif current_page == 4:
                    methods = method_service.get_methods()
                    draw_method_selection(stdscr, methods, current_method, selected_methods)
                    current_page, current_method, selected_methods = handle_method_selection(stdscr, methods, current_method, selected_methods)

                elif current_page == 5:
                    # run evaluation once when entering this page
                    if not run_cli.evaluation_done:
                        stdscr.clear()
                        stdscr.addstr(height // 2, width // 2 - 20, "Running evaluation... Please wait...")
                        stdscr.refresh()
                        
                        selected_methods_enum = [MethodName(method) for method in selected_methods]
                        logging.getLogger().setLevel(logging.ERROR)
                        run_cli.evaluation_result = evaluation_service.evaluate(selected_datasets_enum, selected_classifiers_enum, selected_methods_enum)
                        logging.getLogger().setLevel(logging.DEBUG)
                        run_cli.evaluation_done = True
                    
                    # draw the stored results
                    draw_evaluation_results(stdscr, run_cli.evaluation_result, evaluation_paginated_page)
                    current_page, evaluation_paginated_page = handle_evaluation_results(stdscr, run_cli.evaluation_result, evaluation_paginated_page, len(run_cli.evaluation_result))

                time.sleep(0.05) 

            except curses.error as e:
                logger.error(f"Curses error: {e}")
                show_error_message(stdscr, "Terminal error occurred. Please resize your window.")
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                show_error_message(stdscr, "An unexpected error occurred. Check debug.log for details.")
                time.sleep(2)
                continue

    except Exception as e:
        logger.error(f"Error in CLI: {e}")
        raise

def main():
    """Main entry point with error handling."""
    try:
        curses.wrapper(run_cli)
    except Exception as e:
        logger.error(f"Error in CLI: {e}")
        print(f"Error Details: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())