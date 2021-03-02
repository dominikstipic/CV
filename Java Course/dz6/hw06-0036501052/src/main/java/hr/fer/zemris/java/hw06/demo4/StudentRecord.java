package hr.fer.zemris.java.hw06.demo4;

/**
 * Encapsulation of student information
 * @author Dominik Stipic
 *
 */
public class StudentRecord {
	/**
	 * student's key
	 */
	private String jmbag;
	/**
	 * student's last name
	 */
	private String lastName;
	/**
	 * student's first name
	 */
	private String firstName;
	/**
	 * points from midterm exam
	 */
	private double MI;
	/**
	 * points from final exam
	 */
	private double ZI;
	/**
	 * points from laboratory exercises
	 */
	private double LAB;
	/**
	 * final grade
	 */
	private int grade;
	
	/**
	 * Creates model of <code>Student</code> 
	 * @param jmbag student's key
	 * @param lastName student's last name
	 * @param firstName student's first name
	 * @param mI points from midterm exam
	 * @param zI oints from final exam
	 * @param lAB points from laboratory exercises
	 * @param grade final grade
	 */
	public StudentRecord(String jmbag, String lastName, String firstName, double mI, double zI, double lAB, int grade) {
		this.jmbag = jmbag;
		this.lastName = lastName;
		this.firstName = firstName;
		MI = mI;
		ZI = zI;
		LAB = lAB;
		this.grade = grade;
	}

	/**
	 * gets student jmbag
	 * @return student jmbag
	 */
	public String getJmbag() {
		return jmbag;
	}

	/**
	 * gets student's last name
	 * @return student last name
	 */
	public String getLastName() {
		return lastName;
	}

	/**
	 * gets student's first name
	 * @return student first name
	 */
	public String getFirstName() {
		return firstName;
	}

	/**
	 * gets student's midterm exam points
	 * @return student's midterm exam points
	 */
	public double getMI() {
		return MI;
	}

	/**
	 * gets final exam points
	 * @return final exam points
	 */
	public double getZI() {
		return ZI;
	}

	/**
	 * gets laboratory exercises points
	 * @return lab exercies points
	 */
	public double getLAB() {
		return LAB;
	}

	/**
	 * gets student's final grade 
	 * @return final grade
	 */
	public int getGrade() {
		return grade;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((jmbag == null) ? 0 : jmbag.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		StudentRecord other = (StudentRecord) obj;
		if (jmbag == null) {
			if (other.jmbag != null)
				return false;
		} else if (!jmbag.equals(other.jmbag))
			return false;
		return true;
	}

	public double examSum() {
		return MI + ZI + LAB;
	}
	
	@Override
	public String toString() {
		return jmbag;
	}

	
	
	
	
	
	
}
