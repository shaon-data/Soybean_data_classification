echo off
setlocal EnableDelayedExpansion
echo Checking if need initialization...
SET init=0
if not exist prm (
	SET /P repon=Repository name:
	SET /P repol=Repository link:
	echo "Ripo Link: !repol!"

	mkdir prm
	echo [initialized] > prm/project.log
	echo [remotename:!repon!] >> prm/project.log
	echo [remotelink:!repol!] >> prm/project.log
	echo Queries for sloving Issues > prm/query.txt	
	echo # !repon! > README.md
	echo ## Download >> README.md
	echo For downloading use >> README.md
	echo        `git clone !repol!` >> README.md


	git init
	git add .
	git remote add !repon! !repol!

	SET init=1

)

IF "%init%" == "1" (
	rem pushing starts

	SET /P umes=Commit Message:

	IF [!umes!] == [] (
		echo empty message - con...
		SET umes=continuing
	)


	echo Pushing...
	git add .
	git commit -m "!umes!"
	git push -u !repon! master

	rem pushing ends

	exit
)

echo .....
echo "update" - for updating project in repository
echo "pull" - for updating project in repository
echo "git" - for using git
echo "query" - for using google
echo .....
echo "renamerepon" - Renaming Repo Name
echo "renamerepol" - Renaming Repo Link
echo "remotedel" - Renaming Repo Link


SET /p  command=Give me a Command:

IF "%command%" == "git" (

	SET command=

	echo Retriving project information...
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotename">nul && (
		
			set line=%%a
			set repon=!line:remotename:=!
			Echo Remotename retrived : !repon!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)

	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotelink">nul && (
		
			set line=%%a
			set repon=!line:remotelink:=!
			Echo Remotelink retrived : !repol!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)

	echo .....
	echo "push" - for pushing your update to remote git
	echo "pull" - 
	SET /p command=Give me a Command:

	IF "!command!" == "push" (
		rem push function

		SET command=

		SET /P umes=Commit Message:

		IF [!umes!] == [] (
			echo empty message - con...
			SET umes=continuing
		)

		echo Pushing...
		git add .
		git commit -m "!umes!"
		git push -u !repon! master

		rem push function ends
	)

) ELSE IF "%command%" == "query" (
	SET command=

	SET /p query=Write query, we will pass it to google:
	echo !query! >> prm/query.txt

	set "URLPATH=!query!"
	set "URLPATH=!URLPATH:\=/!"
	set "URLPATH=!URLPATH: =%%20!"
	start https://www.google.com/search?q=!URLPATH!

	endlocal
	endlocal

)ELSE IF "%command%" == "renamerepon" (
	echo Retriving project information...
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotename">nul && (
		
			set line=%%a
			set repon=!line:remotename:=!
			Echo Remotename retrived : !repon!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)

	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotelink">nul && (
		
			set line=%%a
			set repol=!line:remotelink:=!
			Echo Remotelink retrived : !repol!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)
	echo Previous Repository Name: !repon!
	echo Previous Repository Link: !repol!
	SET /P new_repon=New Repository Name:
	git remote rename !repon! !new_repon!
	echo [initialized] > prm/project.log
	echo [remotename:!new_repon!] >> prm/project.log
	echo [remotelink:!repol!] >> prm/project.log

)ELSE IF "%command%" == "renamerepol" (
	echo Retriving project remotename...
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotename">nul && (
		
			set line=%%a
			set repon=!line:remotename:=!
			Echo Remotename retrived : !repon!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotelink">nul && (
		
			set line=%%a
			set repon=!line:remotelink:=!
			Echo Remotelink retrived : !repol!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)
	echo existing Repository Name: !repon!
	echo existing repository link: !repol!
	git remote -v
	SET /P new_repol=New Repository link:
	git remote set-url !repon! !new_repol!
	echo [initialized] > prm/project.log
	echo [remotename:!new_repon!] >> prm/project.log
	echo [remotelink:!new_repol!] >> prm/project.log
)ELSE IF "%command%" == "remotedel" (
	echo Retriving project remotename...
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotename">nul && (
		
			set line=%%a
			set repon=!line:remotename:=!
			Echo Remotename retrived : !repon!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)
	echo Available Remmote names:
	git remote
	SET /P sofake=Remote Name to delete:
	git remote rm !sofake!
	
	IF "%sofake%" == "%repon%" (
		echo [initialized] > prm/project.log
		echo [remotename:] >> prm/project.log
		echo [remotelink:] >> prm/project.log
	)

) ELSE IF "%command%" == "pull" (
	SET command=

	echo Retriving project remotename...
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotename">nul && (
		
			set line=%%a
			set repon=!line:remotename:=!
			Echo Remotename retrived : !repon!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)

	echo Pulling...
	git pull !repon! master	


)ELSE IF "%command%" == "update" (
	SET command=

	echo Retriving project remotename...
	FOR /F "tokens=1* delims=[]" %%a in (prm/project.log) do (
		Echo.%%a | findstr /C:"remotename">nul && (
		
			set line=%%a
			set repon=!line:remotename:=!
			Echo Remotename retrived : !repon!
	
		) || (
		    rem Echo remote name cannot retrive
		)
	)


	SET /P umes=Commit Message:

	IF [!umes!] == [] (
		echo empty message - con...
		SET umes=continuing
	)

	echo Pushing...
	git add .
	git commit -m "!umes!"
	git push -u !repon! master	

)