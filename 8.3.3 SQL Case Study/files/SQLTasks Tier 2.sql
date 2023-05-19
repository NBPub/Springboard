/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 2 of the case study, which means that there'll be less guidance for you about how to setup
your local SQLite connection in PART 2 of the case study. This will make the case study more challenging for you: 
you might need to do some digging, aand revise the Working with Relational Databases in Python chapter in the previous resource.

Otherwise, the questions in the case study are exactly the same as with Tier 1. 

PART 1: PHPMyAdmin
You will complete questions 1-9 below in the PHPMyAdmin interface. 
Log in by pasting the following URL into your browser, and
using the following Username and Password:

URL: https://sql.springboard.com/
Username: <>
Password: <>

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

In this case study, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */


/* QUESTIONS 
/* Q1: Some of the facilities charge a fee to members, but some do not.
Write a SQL query to produce a list of the names of the facilities that do. */
SELECT name FROM Facilities WHERE membercost > 0;


/* Q2: How many facilities do not charge a fee to members? */
SELECT COUNT(*) FROM Facilities WHERE membercost = 0;


/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT facid, name, membercost, monthlymaintenance
FROM Facilities
WHERE membercost < monthlymaintenance*0.2;


/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */
SELECT * FROM Facilities
WHERE facid IN (1,5);


/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */
SELECT name, monthlymaintenance, 
	CASE
		WHEN monthlymaintenance > 100 THEN 'expensive'
		ELSE 'cheap'
	END AS COST
FROM Facilities;


/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */
SELECT firstname, surname FROM Members
WHERE joindate = (SELECT MAX(joindate) FROM Members)


/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */
SELECT name,  surname || ', ' || firstname AS member
FROM Bookings 
LEFT JOIN Facilities USING(facid)
LEFT JOIN Members USING(memid)
WHERE name LIKE 'Tennis Court%'
GROUP BY member
ORDER BY member;


/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */
SELECT name, surname || ', ' || firstname AS member,
	CASE
		WHEN memid = 0  THEN slots*guestcost
		ELSE slots*membercost
	END AS cost
FROM Bookings 
LEFT JOIN Facilities USING(facid)
LEFT JOIN Members USING(memid)
WHERE starttime LIKE '2012-09-14 %'
AND cost > 30
ORDER BY cost DESC;


/* Q9: This time, produce the same result as in Q8, but using a subquery. */
/* get member cost, get guest cost, combine  */
SELECT name, member, slots*perslot AS cost FROM (
	SELECT name, surname || ', ' || firstname AS member, slots, 
		CASE
			WHEN memid > 0  THEN membercost
			ELSE guestcost
		END AS perslot
	FROM Bookings 
	LEFT JOIN Facilities USING(facid)
	LEFT JOIN Members USING(memid)
	WHERE starttime LIKE '2012-09-14 %')
WHERE cost > 30 ORDER BY cost DESC;

/* PART 2: SQLite

Export the country club data from PHPMyAdmin, and connect to a local SQLite instance from Jupyter notebook 
for the following questions.  

QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */
SELECT name, member_rev+guest_rev AS total_revenue FROM 
(SELECT facid, name,  SUM(slots)*membercost AS member_rev
FROM Bookings LEFT JOIN Facilities USING(facid)
WHERE memid > 0 GROUP BY facid)
INNER JOIN (
SELECT facid, SUM(slots)*guestcost AS guest_rev
FROM Bookings LEFT JOIN Facilities USING(facid)
WHERE memid = 0 GROUP BY facid) 
USING(facid) WHERE total_revenue < 1000
ORDER BY total_revenue;


/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */
SELECT M.surname || ', ' || M.firstname AS "Member",
R.surname || ', ' || R.firstname AS "Recommending Member"
FROM Members M
LEFT JOIN Members R on R.memid=M.recommendedby 
WHERE M.memid > 0
ORDER BY Member;


/* Q12: Find the facilities with their usage by member, but not guests */
/* note from me: starttime for bookings in %Y-%m-%d format */
SELECT name AS facility, SUM(slots) AS member_usage FROM Bookings
LEFT JOIN Facilities USING(facid)
WHERE memid > 0 GROUP BY facid ORDER BY member_usage DESC


/* Q13: Find the facilities usage by month, but not guests */
/* All facilities by month */
SELECT SUBSTRING(starttime,6,2) AS month, SUM(slots) AS member_usage 
FROM Bookings
WHERE memid > 0 GROUP BY month;
/* Each facility by month */
SELECT SUBSTRING(starttime,6,2) AS month, name AS facility, SUM(slots) AS member_usage 
FROM Bookings
LEFT JOIN Facilities USING(facid)
WHERE memid > 0 GROUP BY month, facid;
