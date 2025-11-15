import csv
class Writer:
    def __init__(self, filename="alfahack_results.csv"):
        self.filename = filename
        
    def write_answers(self, answers):
        with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='\t')  
            writer.writerow(["q_id", "web_list"])  
            for idx, answer in enumerate(answers, start=1):
                web_list_str = str(answer)
                writer.writerow([idx, web_list_str])

        pass
