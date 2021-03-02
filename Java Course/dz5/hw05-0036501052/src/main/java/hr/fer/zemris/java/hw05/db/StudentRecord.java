package hr.fer.zemris.java.hw05.db;

import java.util.Objects;

/**
 * Enkapsulira sve potebne podatke vezne uz studenta:
 * jmbag,ime,prezime,ocjena
 * @author Dominik Stipic
 *
 */
public class StudentRecord {
	private String JMBAG;
	private String lastName;
	private String firstName;
	private String finalGrade;
	
	/**
	 * Stvara studenta
	 * @param JMBAG jmbag studenta
	 * @param firstName ime studenta
	 * @param lastName prezime studenta
	 * @param finalGrade ocjena studenta
	 * @throws NullPointerException - ako je bilo sto od argumenata null
	 */
	public StudentRecord(String JMBAG, String firstName, String lastName, String finalGrade) {
		this.JMBAG = Objects.requireNonNull(JMBAG, "Jmbag mustnt be null");
		this.lastName = Objects.requireNonNull(lastName, "lastName mustnt be null");
		this.firstName = Objects.requireNonNull(firstName, "firstName mustnt be null");
		this.finalGrade = finalGrade;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((JMBAG == null) ? 0 : JMBAG.hashCode());
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
		if (JMBAG == null) {
			if (other.JMBAG != null)
				return false;
		} else if (!JMBAG.equals(other.JMBAG))
			return false;
		return true;
	}

	
	/**
	 * Getter
	 * @return jmabg
	 */
	public String getJMBAG() {
		return JMBAG;
	}
	

	/**
	 * Getter
	 * @return prezime
	 */
	public String getLastName() {
		return lastName;
	}
	

	/**
	 * Getter
	 * @return ime
	 */
	public String getFirstName() {
		return firstName;
	}
	

	/**
	 * Getter
	 * @return ocjena
	 */
	public String getFinalGrade() {
		return finalGrade;
	}

}
