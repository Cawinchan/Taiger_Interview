from time import sleep
import csv
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys


def find_all_repo_names(browser):
    github_author_and_repo_elements = browser.find_elements_by_xpath('//h3//a[@href]')  # Search for all Authors and Repos elements because they are both refered by the same class
    github_author_elements = browser.find_elements_by_xpath('//h3//a[@href]//span')       # Search for all the Author elements
    author_and_repo_names = [x.text for x in github_author_and_repo_elements]
    author_names = [x.text for x in github_author_elements]
    repo_names = []

    author_and_repo_names = list(zip(author_and_repo_names,author_names))
    for x,y in author_and_repo_names:           # Remove the Author and reformat
            g = x.replace(y+' ', '')
            repo_names.append(g)

    return repo_names


def find_all_star_values(browser):

    stars_elements = browser.find_elements_by_xpath("//a[@class='muted-link d-inline-block mr-3']//*[@class='octicon octicon-star']") #Find star child element of star value as both forked and star share the same class

    stars = []
    for x in stars_elements:
        tmp = x.find_element_by_xpath("..")            # take the star element(which is the parent of the star child element)
        stars.append(tmp.text.replace(",", ""))        # reformat

    return stars


def find_programming_language(browser):
    test = browser.find_elements_by_xpath("//li[@id]")     #finds id element fo authors

    id_lst = []

    for x in test:
        id_lst.append(x.get_attribute('id'))  #Find the id of the authors

    programming_language = []

    for x in id_lst:    #Search if the author has the programminglanguage itemprop, if no, this repo has no programming language
        try:
            y = "//li[@id='"+str(x)+"']//span[@itemprop='programmingLanguage']"
            tmp = browser.find_element_by_xpath(y)
            programming_language.append(tmp.text)
        except NoSuchElementException:    #continues to the next id if there in no programminglanguage itemprop
            tmp = "Null"
            programming_language.append(tmp)
            continue

    return programming_language

def copy_to_csv(name,repo,stars,program):
    github_data = list(zip(repo,stars,program))   #reformating
    print("number of repos", len(github_data))

    with open(str(name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)   #creates a csv file
        writer.writerow(["Github repo name","Number of stars","Programming language"]) #adds in a header for clarity
        writer.writerows((github_data))


    with open('github_trending_data.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)

if __name__ == "__main__":
    browser = webdriver.Firefox()
    browser.get('https://github.com/trending?since=weekly%5D')  # Open up a browser
    sleep(2)
    repo = find_all_repo_names(browser)
    stars = find_all_star_values(browser)
    program = find_programming_language(browser)
    browser.quit()
    copy_to_csv('github_data.csv',repo,stars,program)
