import csv
class Writer:
    def __init__(self, filename="alfahack_results.csv"):
        self.filename = filename
        
    def write_answers(self, answers):
        with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["q_id", "web_list"])
            for q_id, web_list in answers:
                writer.writerow([q_id, str(web_list)])
