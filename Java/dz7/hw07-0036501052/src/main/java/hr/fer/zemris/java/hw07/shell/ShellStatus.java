package hr.fer.zemris.java.hw07.shell;

/**
 * Respresents currents status of the shell.
 * Shell status are: CONTINUE and TERMINATE.
 * When terminate occurs the shell will stop prompting user
 * and finish her job.
 * @author Dominik Stipić
 *
 */
public enum ShellStatus {
	/**
	 * Shell continues her work
	 */
	CONTINUE,
	/**
	 * Shell stops her work 
	 */
	TERMINATE
}
