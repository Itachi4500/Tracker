CREATE TABLE employees (
    id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    position VARCHAR(255) NOT NULL,
    department VARCHAR(255) NOT NULL,
    dob DATE NOT NULL
);

CREATE TABLE attendance (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id VARCHAR(20),
    check_in DATETIME,
    check_out DATETIME,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);
