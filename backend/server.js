// server.js

const express = require('express');
const mysql = require('mysql2/promise');
const cors = require('cors');

const app = express();
const port = 3000;

// Middleware
app.use(cors()); // Allow requests from our frontend
app.use(express.json()); // Allow the server to understand JSON data

// --- DATABASE CONNECTION ---
// IMPORTANT: Replace these with your actual database credentials.
// It's recommended to use environment variables for security.
const dbConfig = {
    host: 'localhost',
    user: 'postgres',       // e.g., 'root'
    password: '1206', // e.g., 'password'
    database: 'attendance_db'   // The database name from your .sql file
};

// Create a connection pool for efficient database connections
const pool = mysql.createPool(dbConfig);


// --- API ROUTES (ENDPOINTS) ---

/**
 * @api {get} /api/employees Get All Employees and Their Attendance
 * @description Optimized to fetch all employees and their attendance records with a single LEFT JOIN query
 * to avoid the N+1 problem.
 */
app.get('/api/employees', async (req, res) => {
    try {
        const query = `
            SELECT 
                e.id, e.name, e.position, e.department, e.dob,
                a.record_id, a.check_in, a.check_out
            FROM 
                employees e
            LEFT JOIN 
                attendance a ON e.id = a.employee_id
            ORDER BY 
                e.name, a.check_in DESC;
        `;
        const [rows] = await pool.query(query);
        
        // Process the flat SQL results into a nested JSON structure
        const employeesMap = new Map();
        rows.forEach(row => {
            if (!employeesMap.has(row.id)) {
                employeesMap.set(row.id, {
                    id: row.id,
                    name: row.name,
                    position: row.position,
                    department: row.department,
                    dob: row.dob,
                    attendance: []
                });
            }

            if (row.record_id) { // Check if there is an attendance record
                employeesMap.get(row.id).attendance.push({
                    record_id: row.record_id,
                    date: row.check_in ? new Date(row.check_in).toISOString().split('T')[0] : null,
                    checkIn: row.check_in ? new Date(row.check_in).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) : null,
                    checkOut: row.check_out ? new Date(row.check_out).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) : null
                });
            }
        });
        
        res.json(Array.from(employeesMap.values()));

    } catch (error) {
        console.error("Failed to fetch employees:", error);
        res.status(500).json({ message: 'Failed to fetch data from the database.' });
    }
});

/**
 * @api {post} /api/employees Add a New Employee
 */
app.post('/api/employees', async (req, res) => {
    try {
        const { id, name, position, department, dob } = req.body;
        if (!id || !name || !position || !department || !dob) {
            return res.status(400).json({ message: 'All fields are required.' });
        }
        const sql = "INSERT INTO employees (id, name, position, department, dob) VALUES (?, ?, ?, ?, ?)";
        await pool.query(sql, [id, name, position, department, dob]);
        res.status(201).json({ message: 'Employee added successfully' });
    } catch (error) {
        console.error("Failed to add employee:", error);
        res.status(500).json({ message: 'Database error while adding employee.' });
    }
});

/**
 * @api {put} /api/employees/:id Update an Existing Employee
 * @description NEWLY ADDED: This endpoint allows editing an employee's details.
 */
app.put('/api/employees/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const { name, position, department, dob } = req.body;
        if (!name || !position || !department || !dob) {
            return res.status(400).json({ message: 'All fields are required.' });
        }
        const sql = "UPDATE employees SET name = ?, position = ?, department = ?, dob = ? WHERE id = ?";
        const [result] = await pool.query(sql, [name, position, department, dob, id]);

        if (result.affectedRows === 0) {
            return res.status(404).json({ message: 'Employee not found.' });
        }
        res.json({ message: 'Employee updated successfully' });
    } catch (error) {
        console.error("Failed to update employee:", error);
        res.status(500).json({ message: 'Database error while updating employee.' });
    }
});


/**
 * @api {delete} /api/employees/:id Delete an Employee
 */
app.delete('/api/employees/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const [result] = await pool.query("DELETE FROM employees WHERE id = ?", [id]);

        if (result.affectedRows === 0) {
            return res.status(404).json({ message: 'Employee not found.' });
        }
        res.json({ message: 'Employee deleted successfully' });
    } catch (error) {
        console.error("Failed to delete employee:", error);
        res.status(500).json({ message: 'Database error while deleting employee.' });
    }
});

/**
 * @api {post} /api/attendance/:employeeId Check-in or Check-out an Employee
 * @description Handles both check-in and check-out logic based on the employee's last status for the day.
 */
app.post('/api/attendance/:employeeId', async (req, res) => {
    try {
        const { employeeId } = req.params;
        const today = new Date().toISOString().split('T')[0];

        // Find the last record for today to see if we are checking in or out
        const [lastRecord] = await pool.query(
            "SELECT * FROM attendance WHERE employee_id = ? AND DATE(check_in) = ? ORDER BY check_in DESC LIMIT 1",
            [employeeId, today]
        );

        if (lastRecord.length === 0 || lastRecord[0].check_out) {
            // CASE 1: No record found for today, OR the last record is already checked out.
            // ACTION: Create a new record to CHECK IN the employee.
            await pool.query("INSERT INTO attendance (employee_id, check_in) VALUES (?, NOW())", [employeeId]);
            res.json({ message: 'Checked In' });
        } else {
            // CASE 2: The last record for today exists and is NOT checked out.
            // ACTION: Update the existing record to CHECK OUT the employee.
            await pool.query("UPDATE attendance SET check_out = NOW() WHERE record_id = ?", [lastRecord[0].record_id]);
            res.json({ message: 'Checked Out' });
        }
    } catch (error) {
        console.error("Failed to update attendance:", error);
        res.status(500).json({ message: 'Database error while updating attendance.' });
    }
});


// Start the server
app.listen(port, () => {
    console.log(`Backend server running at http://localhost:${port}`);
});
