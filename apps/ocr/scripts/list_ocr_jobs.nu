#!/usr/bin/env nu

ocr list --limit 1000
	| from json 
	| get jobs 
	| where file_path =~ "tenth" 
	| each { |r| $r | update file_path ($r.file_path | path basename) } 
	| sort-by file_path 
	| select file_path document_id status created_at




