@example @web @quotes
Feature: Inspira inspirational quote page
  Inspira serves a web page that shows one inspirational quote chosen from a local text file.
  Each browser refresh triggers a fresh selection attempt.

  Background:
    Given the quote file path is configured

  @critical
  Scenario: GET / returns one quote from the catalog
    Given the quote file contains:
      """
      Stay hungry, stay foolish.
      The only way out is through.
      Fortune favors the bold.
      """
    And the random index source returns 1
    When the browser requests "/"
    Then the response status is 200
    And the response content type is "text/html"
    And the page displays the quote "The only way out is through."

  @critical
  Scenario: Each refresh performs a fresh selection attempt
    Given the quote file contains:
      """
      Stay hungry, stay foolish.
      The only way out is through.
      Fortune favors the bold.
      """
    And the random index source returns 2, then 0
    When the browser requests "/" twice
    Then the first response displays "Fortune favors the bold."
    And the second response displays "Stay hungry, stay foolish."

  @standard
  Scenario: A refresh is not required to show a different quote
    Given the quote file contains:
      """
      Stay hungry, stay foolish.
      The only way out is through.
      """
    And the random index source returns 0, then 0
    When the browser requests "/" twice
    Then both responses display "Stay hungry, stay foolish."

  @standard
  Scenario: Blank lines are ignored and surrounding whitespace is trimmed
    Given the quote file contains:
      """

        Stay hungry, stay foolish.

      The only way out is through.   
      """
    And the random index source returns 1
    When the browser requests "/"
    Then the page displays the quote "The only way out is through."

  @critical
  Scenario: Quote text is escaped before rendering
    Given the quote file contains:
      """
      <script>alert("x")</script>
      """
    And the random index source returns 0
    When the browser requests "/"
    Then the response status is 200
    And the page displays the text "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"
    And the page does not execute script markup from the quote file

  @critical @error-handling
  Scenario: Missing quote file returns a configuration error page
    Given the configured quote file does not exist
    When the browser requests "/"
    Then the response status is 503
    And the page displays "Quotes are temporarily unavailable."
    And no quote is displayed

  @critical @error-handling
  Scenario: A file with no valid quotes returns a configuration error page
    Given the quote file contains:
      """


      """
    When the browser requests "/"
    Then the response status is 503
    And the page displays "Quotes are temporarily unavailable."
    And no quote is displayed

  @standard
  Scenario: Updating the quote file affects the next refresh
    Given the quote file contains:
      """
      Stay hungry, stay foolish.
      """
    And the random index source returns 0, then 1
    When the browser requests "/"
    And the quote file is replaced with:
      """
      Stay hungry, stay foolish.
      The only way out is through.
      """
    And the browser requests "/" again
    Then the second response displays "The only way out is through."
