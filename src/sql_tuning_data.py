import json

schema = {
   "department": ["deptid", "deptname", "location", "name", "head_id", "head_age"],
   "teacher": ["teacherid", "firstname", "lastname", "email", "deptid"],
   "subject": ["subjectid", "subjectname", "credits", "teacherid"],
   "student": ["studentid", "firstname", "lastname", "dob", "deptid"],
   "enrollment": ["enrollmentid", "studentid", "subjectid", "grade", "semester"]
}

# Your training_prompts list goes here (omitted for brevity)
training_prompts = [
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "department (deptid, deptname, location, id, name, head_id, head_age); teacher (teacherid, firstname, lastname, email, deptid); subject (subjectid, subjectname, credits, teacherid); student (studentid, firstname, lastname, dob, deptid); enrollment (enrollmentid, studentid, subjectid, grade, semester)",
    "question": "List the names of all departments located in 'Building A'.",
    "answer": "SELECT deptname FROM department WHERE location = 'Building A';"
  },
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "department (deptid...); teacher (teacherid...); student (studentid...);",
    "question": "Show the first and last names of students born after January 1st, 2005.",
    "answer": "SELECT firstname, lastname FROM student WHERE dob > '2005-01-01';"
  },
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "teacher (teacherid, firstname, lastname, deptid); department (deptid, deptname);",
    "question": "How many teachers are in the 'Mathematics' department?",
    "answer": "SELECT COUNT(*) FROM teacher JOIN department ON teacher.deptid = department.deptid WHERE department.deptname = 'Mathematics';"
  },
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "subject (subjectid, subjectname, credits, teacherid); teacher (teacherid, firstname, lastname);",
    "question": "List all subject names along with the last name of the teacher who teaches them.",
    "answer": "SELECT s.subjectname, t.lastname FROM subject s JOIN teacher t ON s.teacherid = t.teacherid;"
  },
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "enrollment (studentid, grade); student (studentid, firstname, lastname);",
    "question": "Find the first and last names of students who received an 'A' grade in any semester.",
    "answer": "SELECT DISTINCT s.firstname, s.lastname FROM student s JOIN enrollment e ON s.studentid = e.studentid WHERE e.grade = 'A';"
  },
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "department (head_age, deptname);",
    "question": "What is the average age of the department heads?",
    "answer": "SELECT AVG(head_age) FROM department;"
  },
  {
    "instruction": "Generate a SQL query based on the provided schema to answer the user's question.",
    "schema": "subject (credits);",
    "question": "Which subjects provide more than 3 credits?",
    "answer": "SELECT subjectname FROM subject WHERE credits > 3;"
  },
  {
    "question": "Which students are enrolled in subjects taught by 'Dr. Smith'?",
    "answer": "SELECT DISTINCT s.firstname, s.lastname FROM student s JOIN enrollment e ON s.studentid = e.studentid JOIN subject sub ON e.subjectid = sub.subjectid JOIN teacher t ON sub.teacherid = t.teacherid WHERE t.lastname = 'Smith';"
  },
  {
    "question": "Find all subjects and their credit counts for students in the 'Physics' department.",
    "answer": "SELECT DISTINCT sub.subjectname, sub.credits FROM subject sub JOIN enrollment e ON sub.subjectid = e.subjectid JOIN student s ON e.studentid = s.studentid JOIN department d ON s.deptid = d.deptid WHERE d.deptname = 'Physics';"
  },
{
    "question": "Show the average grade for each subject in the 'Spring' semester.",
    "answer": "SELECT subjectid, AVG(CAST(grade AS FLOAT)) FROM enrollment WHERE semester = 'Spring' GROUP BY subjectid;"
  },
  {
    "question": "List departments that have more than 50 students enrolled.",
    "answer": "SELECT d.deptname FROM department d JOIN student s ON d.deptid = s.deptid GROUP BY d.deptid, d.deptname HAVING COUNT(s.studentid) > 50;"
  },
 {
    "question": "Who are the top 3 oldest students in the Computer Science department?",
    "answer": "SELECT firstname, lastname, dob FROM student s JOIN department d ON s.deptid = d.deptid WHERE d.deptname = 'Computer Science' ORDER BY dob ASC LIMIT 3;"
  },
  {
    "question": "Get all teachers whose email addresses end with '@university.edu'.",
    "answer": "SELECT firstname, lastname FROM teacher WHERE email LIKE '%@university.edu';"
  },
{
    "question": "Find students who have not enrolled in any subjects yet.",
    "answer": "SELECT firstname, lastname FROM student WHERE studentid NOT IN (SELECT studentid FROM enrollment);"
  },
  {
    "question": "Identify departments where the head's age is greater than the average head age of all departments.",
    "answer": "SELECT name FROM department WHERE head_age > (SELECT AVG(head_age) FROM department);"
  }
]


def format_schema(schema_dict):
    """Converts schema dict to a readable string for the prompt."""
    return " | ".join([f"{table} ({', '.join(cols)})" for table, cols in schema_dict.items()])

def generate_tuning_data(prompts, schema_dict, output_file):
    formatted_data = []
    schema_str = format_schema(schema_dict)

    for item in prompts:
        # Construct the instruction-based chat object
        chat_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a SQL expert. Return only the SQL query based on the schema provided."
                },
                {
                    "role": "user",
                    "content": f"Schema: {schema_str}\nQuestion: {item['question']}"
                },
                {
                    "role": "assistant",
                    "content": item['answer']
                }
            ]
        }
        formatted_data.append(chat_entry)

    # Save as JSONL (Standard for Fine-tuning)
    with open(output_file, 'w') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Success! {len(formatted_data)} training rows saved to {output_file}")

# Execute
generate_tuning_data(training_prompts, schema, "sql_tuning_data.jsonl")
